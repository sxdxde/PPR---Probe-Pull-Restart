"""
Experiment 1b: does anything fix the probing-dimensionality collapse
found in exp1_rastrigin.py (see dimensionality_collapse.png / RESULTS.md)?

Extends the original Rastrigin dimension sweep past 20-D up to 15,000-D
(the brief's ">10k parameters" bar) and tests three cheap fixes, in the
order specified in CLAUDE_CODE_BRIEF.md Task 1:

  (a) subset      -- restrict probing to a small, FIXED-size subset of
                      coordinates (probe_subset_size=8), regardless of
                      total dimension n. Generalizes GDSolver's
                      "last-layer-only" restriction to a vector with no
                      layers: probe the last k coordinates only.
  (b) scaled_dirs -- keep full-dimensional isotropic probing, but scale
                      the direction budget with dimension:
                      probe_dirs = min(64, 4*sqrt(n)).
  (c) grad_pca    -- probe along the true top singular vectors of the
                      last `grad_history` gradients (via SVD), not
                      random linear combinations of them (that was
                      already tried and failed -- see RESULTS.md #4).

`baseline` (probe_dirs=8, full random directions, no fix) is included
at every dimension as the control -- this is exactly the config that
produced dimensionality_collapse.png.

Metric: probe success rate (successful pulls / total probes attempted),
summed across seeds, at each dimension -- the same metric and the same
plot shape as the original.
"""
import math
import json
import time
import torch
from ppr_optimizer import PPR

A = 10.0
DOMAIN = 5.12
STEPS = 3000
SEEDS = list(range(5))
DIMS = [2, 5, 20, 100, 1000, 5000, 15000]

MOM_LR = 0.002


def rastrigin(x):
    return A * x.numel() + (x**2 - A * torch.cos(2 * math.pi * x)).sum()


def init_x(dim, seed):
    torch.manual_seed(seed)
    return torch.nn.Parameter((torch.rand(dim) * 2 - 1) * DOMAIN)


def run_baseline_momentum(dim, seed, lr=MOM_LR, steps=STEPS):
    x = init_x(dim, seed)
    opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
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


def run_ppr(dim, seed, ppr_kwargs, steps=STEPS):
    x = init_x(dim, seed)
    opt = PPR(
        [x], lr=MOM_LR, momentum=0.9,
        tentacles=True, reincarnation=True,
        reach_abs=1.8, pull_force=0.15, pull_steps=40,
        accept_tau=1e-2, progress_tau=5e-2,
        stall_eps=1e-4, stall_steps=25, settle_steps=15,
        max_attempts=5, jitter=1e-3,
        **ppr_kwargs,
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


def fix_kwargs(name, dim):
    if name == "baseline":
        return dict(probe_dirs=8, probe_mode="random")
    if name == "subset":
        return dict(probe_dirs=8, probe_mode="random", probe_subset_size=8)
    if name == "scaled_dirs":
        pd = min(64, max(1, int(4 * math.sqrt(dim))))
        return dict(probe_dirs=pd, probe_mode="random")
    if name == "grad_pca":
        return dict(probe_dirs=20, probe_mode="grad_pca", grad_history=10)
    raise ValueError(name)


def stat(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=mean, std=var ** 0.5)


def main():
    t0 = time.time()
    fixes = ["baseline", "subset", "scaled_dirs", "grad_pca"]
    results = {fix: {} for fix in fixes}

    for dim in DIMS:
        print(f"\n=== dimension {dim} ===")
        mom_vals = [run_baseline_momentum(dim, s) for s in SEEDS]
        mom_stat = stat(mom_vals)
        print(f"  plain_momentum: {mom_stat}")

        for fix in fixes:
            kwargs = fix_kwargs(fix, dim)
            losses, stats_list = [], []
            beats_momentum = 0
            for s in SEEDS:
                v, st = run_ppr(dim, s, kwargs)
                losses.append(v)
                stats_list.append(st)
                if math.isfinite(v) and math.isfinite(mom_vals[s]) and v < mom_vals[s] - 1e-2:
                    beats_momentum += 1
            total_probes = sum(st["probes"] for st in stats_list)
            total_success = sum(st["successful_pulls"] for st in stats_list)
            rate = (100.0 * total_success / total_probes) if total_probes else 0.0
            results[fix][dim] = dict(
                loss=stat(losses),
                probe_dirs=kwargs.get("probe_dirs"),
                total_probes=total_probes,
                total_success=total_success,
                success_rate_pct=rate,
                beats_momentum=f"{beats_momentum}/{len(SEEDS)}",
            )
            print(f"  {fix:12s} probe_dirs={kwargs.get('probe_dirs'):>3} "
                  f"probes={total_probes:>5} success={total_success:>4} "
                  f"rate={rate:5.2f}%  loss_mean={results[fix][dim]['loss']['mean']:.3f} "
                  f"beats_mom={results[fix][dim]['beats_momentum']}")

    with open("exp1b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
