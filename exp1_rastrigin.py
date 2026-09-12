"""
Experiment 1: synthetic non-convex stress test -- a dimension sweep.

Rastrigin function, dimension n in {2, 5, 20}. This directly tests the
"dimensionality of probing" limitation flagged in Section 8 of the
proposal: PPR's random-direction probe uses a small, FIXED number of
directions (k=8) regardless of dimension, because the whole point of
gating it behind the stall detector is to keep it cheap. The prediction
under test is that this fixed budget should work reasonably at low
dimension and degrade as dimension grows, because a fixed number of
random directions covers a shrinking fraction of the space.

For each dimension we report, across 8 seeds:
  - plain momentum's converged ("stuck") loss  -- the baseline PPR must beat
  - Adam / AdamW's converged loss
  - PPR (full) 's best loss, and how often it *beats* plain momentum's
    stuck value on the SAME seed (i.e. same starting point) -- this is
    the direct, per-seed test of whether tentacles+reincarnation add
    anything over the base optimizer they are built on top of.
"""
import math
import json
import time
import torch
from ppr_optimizer import PPR

A = 10.0
DOMAIN = 5.12
STEPS = 3000
SEEDS = list(range(8))
DIMS = [2, 5, 20]

# learning rates chosen so plain momentum SETTLES (does not oscillate
# chaotically) -- verified empirically before this script was written.
MOM_LR = 0.002
ADAM_LR = 0.05


def rastrigin(x):
    return A * x.numel() + (x**2 - A * torch.cos(2 * math.pi * x)).sum()


def init_x(dim, seed):
    torch.manual_seed(seed)
    return torch.nn.Parameter((torch.rand(dim) * 2 - 1) * DOMAIN)


def run_baseline(kind, dim, seed, lr, steps=STEPS):
    x = init_x(dim, seed)
    if kind == "sgd_momentum":
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
    elif kind == "adam":
        opt = torch.optim.Adam([x], lr=lr)
    elif kind == "adamw":
        opt = torch.optim.AdamW([x], lr=lr)
    best = float("inf")
    for _ in range(steps):
        opt.zero_grad()
        loss = rastrigin(x)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            return float("nan")
        best = min(best, loss.item())
    return best


def run_ppr(dim, seed, tentacles, reincarnation, steps=STEPS):
    x = init_x(dim, seed)
    opt = PPR(
        [x], lr=MOM_LR, momentum=0.9,
        tentacles=tentacles, reincarnation=reincarnation,
        probe_dirs=8, reach_abs=1.8, pull_force=0.15, pull_steps=40,
        accept_tau=1e-2, progress_tau=5e-2,
        stall_eps=1e-4, stall_steps=25, settle_steps=15,
        max_attempts=5, jitter=1e-3,
    )

    def closure():
        opt.zero_grad()
        loss = rastrigin(x)
        loss.backward()
        return loss

    def loss_only():
        with torch.no_grad():
            return rastrigin(x).item()

    for _ in range(steps):
        loss = opt.step(closure, loss_only)
        if not math.isfinite(loss.item()):
            return float("nan"), opt.stats()
    return opt.g["best_loss"], opt.stats()


def stat(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=mean, std=var ** 0.5)


def main():
    t0 = time.time()
    results = {}

    for dim in DIMS:
        print(f"\n=== dimension {dim} ===")
        mom_vals = [run_baseline("sgd_momentum", dim, s, MOM_LR) for s in SEEDS]
        adam_vals = [run_baseline("adam", dim, s, ADAM_LR) for s in SEEDS]
        adamw_vals = [run_baseline("adamw", dim, s, ADAM_LR) for s in SEEDS]

        ppr_full, ppr_stats = [], []
        ppr_tent_only = []
        ppr_reinc_only = []
        beats_momentum = 0
        for s in SEEDS:
            v, st = run_ppr(dim, s, True, True)
            ppr_full.append(v)
            ppr_stats.append(st)
            if math.isfinite(v) and math.isfinite(mom_vals[s]) and v < mom_vals[s] - 1e-2:
                beats_momentum += 1
            v2, _ = run_ppr(dim, s, True, False)
            ppr_tent_only.append(v2)
            v3, _ = run_ppr(dim, s, False, True)
            ppr_reinc_only.append(v3)

        total_probes = sum(st["probes"] for st in ppr_stats)
        total_success = sum(st["successful_pulls"] for st in ppr_stats)

        results[dim] = dict(
            sgd_momentum=stat(mom_vals), adam=stat(adam_vals), adamw=stat(adamw_vals),
            ppr_full=stat(ppr_full), ppr_tentacles_only=stat(ppr_tent_only),
            ppr_reincarnation_only=stat(ppr_reinc_only),
            ppr_beats_momentum_same_seed=f"{beats_momentum}/{len(SEEDS)}",
            ppr_total_probes=total_probes, ppr_successful_pulls=total_success,
            ppr_probe_success_rate=f"{total_success}/{total_probes}" if total_probes else "0/0",
        )
        for k, v in results[dim].items():
            print(f"  {k}: {v}")

    with open("exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
