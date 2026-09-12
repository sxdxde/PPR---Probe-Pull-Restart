"""
Experiment 1c: axis-aligned coordinate probing -- the structural fix.

Context: exp1b tested three fixes for the dimensionality collapse and all
failed. Root-cause analysis revealed the shared failure mode: every prior
fix still probed directions that move MULTIPLE coordinates simultaneously.
The probability of all k subset coordinates simultaneously landing in better
basins is (p_1D)^k -- exponentially small even for k=8.

This experiment tests two new probe_mode variants that move exactly ONE
coordinate per probe:

  (d) coord_align           -- probe ±reach along each coordinate axis in a
                               fixed subset of 8 coordinates (last 8 params).
                               Total probes per stall: 2*8 = 16.
                               Key property: success probability per probe ≈
                               p_1D, independent of total parameter count n.

  (e) coord_align_r1        -- same but reach_abs=1.0 (matched to Rastrigin's
                               unit basin spacing instead of the default 1.8).
                               Tests whether reach tuning matters independently
                               of the structural axis-alignment fix.

  (f) coord_align_multi     -- ±{0.5,1.0,1.5,2.0}×reach per coordinate.
                               Removes reach sensitivity by sweeping 4 scales.
                               Total probes per stall: 4*2*8 = 64.

Baseline and the three previously-rejected fixes are NOT re-run here (their
0% at 15k-D is already documented in RESULTS.md and exp1b_results.json).
The baseline is included at each dimension only as the control reference point.

Same sweep as exp1b: DIMS = [2, 5, 20, 100, 1000, 5000, 15000], 5 seeds each.
"""
import math
import json
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run_baseline_momentum(dim, seed, steps=STEPS):
    x = init_x(dim, seed)
    opt = torch.optim.SGD([x], lr=MOM_LR, momentum=0.9)
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
    kwargs = dict(ppr_kwargs)
    reach_abs = kwargs.pop("reach_abs", 1.8)
    opt = PPR(
        [x], lr=MOM_LR, momentum=0.9,
        tentacles=True, reincarnation=True,
        reach_abs=reach_abs, pull_force=0.15, pull_steps=40,
        accept_tau=1e-2, progress_tau=5e-2,
        stall_eps=1e-4, stall_steps=25, settle_steps=15,
        max_attempts=5, jitter=1e-3,
        **kwargs,
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


# Each entry: (label, fix_kwargs_for_ppr_constructor)
FIXES = [
    ("baseline",           dict(probe_dirs=8, probe_mode="random")),
    ("coord_align_s8",     dict(probe_mode="coord_align",            probe_subset_size=8)),
    ("coord_align_s8_r1",  dict(probe_mode="coord_align",            probe_subset_size=8, reach_abs=1.0)),
    ("coord_align_multi",  dict(probe_mode="coord_align_multi_reach", probe_subset_size=8)),
]


def stat(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=mean, std=var ** 0.5)


def main():
    t0 = time.time()
    results = {fix: {} for fix, _ in FIXES}

    for dim in DIMS:
        print(f"\n=== dimension {dim} ===")
        mom_vals = [run_baseline_momentum(dim, s) for s in SEEDS]
        mom_stat = stat(mom_vals)
        print(f"  plain_momentum: mean={mom_stat['mean']:.2f}")

        for fix_name, fix_kw in FIXES:
            losses, stats_list = [], []
            beats_momentum = 0
            for s in SEEDS:
                v, st = run_ppr(dim, s, dict(fix_kw))
                losses.append(v)
                stats_list.append(st)
                if math.isfinite(v) and math.isfinite(mom_vals[s]) and v < mom_vals[s] - 1e-2:
                    beats_momentum += 1
            total_probes = sum(st["probes"] for st in stats_list)
            total_success = sum(st["successful_pulls"] for st in stats_list)
            rate = (100.0 * total_success / total_probes) if total_probes else 0.0
            results[fix_name][dim] = dict(
                loss=stat(losses),
                total_probes=total_probes,
                total_success=total_success,
                success_rate_pct=rate,
                beats_momentum=f"{beats_momentum}/{len(SEEDS)}",
            )
            print(f"  {fix_name:20s} probes={total_probes:>5} success={total_success:>4} "
                  f"rate={rate:6.2f}%  loss={results[fix_name][dim]['loss']['mean']:.2f}  "
                  f"beats_mom={results[fix_name][dim]['beats_momentum']}")

    with open("exp1c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")

    _plot(results)


def _plot(results):
    dims = DIMS
    colors = {"baseline": "gray", "coord_align_s8": "tab:blue",
              "coord_align_s8_r1": "tab:orange", "coord_align_multi": "tab:green"}
    labels = {"baseline": "baseline (random, 8 dirs)",
              "coord_align_s8": "(d) coord_align reach=1.8",
              "coord_align_s8_r1": "(e) coord_align reach=1.0",
              "coord_align_multi": "(f) coord_align multi-reach"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for fix_name, _ in FIXES:
        rates = [results[fix_name].get(d, {}).get("success_rate_pct", 0.0) for d in dims]
        ax1.plot(dims, rates, marker="o", label=labels[fix_name], color=colors[fix_name])

    ax1.set_xscale("log")
    ax1.set_xlabel("Parameter count (dimension)")
    ax1.set_ylabel("Probe success rate (%)")
    ax1.set_title("Probe success rate vs. dimension\n(coord_align fixes vs. baseline)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    for fix_name, _ in FIXES:
        losses = [results[fix_name].get(d, {}).get("loss", {}).get("mean", float("nan")) for d in dims]
        ax2.plot(dims, losses, marker="o", label=labels[fix_name], color=colors[fix_name])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Parameter count (dimension)")
    ax2.set_ylabel("Best loss reached (log scale)")
    ax2.set_title("Best loss reached vs. dimension")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coord_align_results.png", dpi=130)
    print("Saved coord_align_results.png")


if __name__ == "__main__":
    main()
