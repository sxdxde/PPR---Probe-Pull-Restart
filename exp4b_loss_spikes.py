"""
Experiment 4b: Reincarnation under injected loss spikes.

The proposal (§Phase 4) motivates reincarnation with the PaLM training
incident: a sudden loss spike from which the optimizer must recover.
Reincarnation's job is to detect that the current position is worse
than the best checkpoint, and restore it. This scenario has never been
tested in any prior experiment.

Setup:
  Dataset: sklearn digits, full-batch (1347 train / 450 test).
  Model: Linear(64,32)->ReLU->Linear(32,10), 2,410 params.
  Training: 250 steps. Spike injection: every 50 steps (steps 50, 100,
  150, 200), one gradient step is computed with fully randomized labels.
  This forces a large, random gradient update that destabilizes velocity.

  Conditions (5 seeds each):
    plain_momentum    -- SGD+momentum, no PPR. Baseline recovery speed.
    reinc_only        -- PPR with reincarnation enabled, no probing.
                         Reincarnation fires after max_attempts stall
                         failures -- tests whether it helps recovery.
    ppr_full          -- coord_align probing + reincarnation.

  Metrics per condition:
    - Loss trajectory (sampled every step) → stored and plotted
    - Final test accuracy
    - Reincarnation count (for PPR conditions)
    - Recovery time per spike: steps until true-label loss returns to
      within 10% of the pre-spike loss value.

  stall detection for PPR: stall_eps=0.01, stall_steps=15.
  Under normal digits training the loss converges and velocity settles,
  so stalls fire naturally as the model converges. After a spike, the
  optimizer either recovers on its own or stalls at a worse position,
  triggering reincarnation.
"""
import math
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ppr_optimizer import PPR

SEEDS = [0, 1, 2, 3, 4]
STEPS = 250
SPIKE_EVERY = 50        # steps 50, 100, 150, 200
RECOVERY_THRESHOLD = 0.10  # within 10% of pre-spike loss = recovered


def get_data(seed):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def eval_acc(model, Xte, yte):
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


def recovery_steps(loss_curve, spike_steps, threshold=RECOVERY_THRESHOLD):
    """For each spike, count steps until loss returns within threshold of pre-spike loss."""
    recoveries = []
    for sp in spike_steps:
        if sp <= 0 or sp >= len(loss_curve):
            continue
        pre = loss_curve[sp - 1]
        if not math.isfinite(pre) or pre <= 0:
            continue
        target = pre * (1 + threshold)
        recovered = None
        for i in range(sp, min(sp + SPIKE_EVERY, len(loss_curve))):
            if loss_curve[i] <= target:
                recovered = i - sp
                break
        recoveries.append(recovered if recovered is not None else SPIKE_EVERY)
    return recoveries


def run_plain_momentum(seed, lr=0.1):
    Xtr, ytr, Xte, yte = get_data(seed)
    model = make_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_curve = []
    n_classes = 10
    for step in range(STEPS):
        is_spike = (step > 0 and step % SPIKE_EVERY == 0)
        y_use = torch.randint(0, n_classes, ytr.shape) if is_spike else ytr
        opt.zero_grad()
        loss = crit(model(Xtr), y_use)
        loss.backward()
        opt.step()
        with torch.no_grad():
            true_loss = crit(model(Xtr), ytr).item()
        loss_curve.append(true_loss if math.isfinite(true_loss) else float("nan"))
    return eval_acc(model, Xte, yte), loss_curve


def run_ppr_condition(seed, tentacles, reincarnation, lr=0.1):
    Xtr, ytr, Xte, yte = get_data(seed)
    model = make_model(seed)
    crit = nn.CrossEntropyLoss()
    n_classes = 10

    ppr_kwargs = dict(
        lr=lr, momentum=0.9,
        tentacles=tentacles, reincarnation=reincarnation,
        reach_frac=0.05, pull_force=0.1, pull_steps=5,
        accept_tau=0.01, progress_tau=0.05,
        stall_eps=0.01, stall_steps=15, settle_steps=5,
        max_attempts=3, jitter=1e-4,
    )
    if tentacles:
        ppr_kwargs["probe_mode"] = "coord_align"
        ppr_kwargs["probe_subset_size"] = 330  # full last layer

    opt = PPR(model.parameters(), **ppr_kwargs)
    loss_curve = []
    spike_active = [False]
    y_spike = [None]

    def closure():
        opt.zero_grad()
        y_use = y_spike[0] if spike_active[0] else ytr
        loss = crit(model(Xtr), y_use)
        loss.backward()
        return loss

    def loss_only():
        with torch.no_grad():
            return crit(model(Xtr), ytr).item()

    for step in range(STEPS):
        spike_active[0] = (step > 0 and step % SPIKE_EVERY == 0)
        if spike_active[0]:
            y_spike[0] = torch.randint(0, n_classes, ytr.shape)
        loss = opt.step(closure, loss_only)
        with torch.no_grad():
            true_loss = crit(model(Xtr), ytr).item()
        loss_curve.append(true_loss if math.isfinite(true_loss) else float("nan"))

    return eval_acc(model, Xte, yte), loss_curve, opt.stats()


def stat(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return mean, var ** 0.5


def main():
    t0 = time.time()
    spike_steps = [s for s in range(STEPS) if s > 0 and s % SPIKE_EVERY == 0]
    print("=" * 64)
    print(f"LOSS SPIKE RECOVERY  ({STEPS} steps, spikes at {spike_steps})")
    print("=" * 64)

    conditions = {
        "plain_momentum": [],
        "reinc_only": [],
        "ppr_full": [],
    }
    curves = {k: [] for k in conditions}
    reinc_counts = {"reinc_only": [], "ppr_full": []}
    accs = {k: [] for k in conditions}

    for seed in SEEDS:
        print(f"\n  seed={seed}")

        acc, lc = run_plain_momentum(seed)
        accs["plain_momentum"].append(acc)
        curves["plain_momentum"].append(lc)
        rec = recovery_steps(lc, spike_steps)
        print(f"    plain_momentum  acc={acc*100:.1f}%  "
              f"recovery steps (per spike): {rec}")

        acc, lc, st = run_ppr_condition(seed, tentacles=False, reincarnation=True)
        accs["reinc_only"].append(acc)
        curves["reinc_only"].append(lc)
        reinc_counts["reinc_only"].append(st["reincarnations"])
        rec = recovery_steps(lc, spike_steps)
        print(f"    reinc_only      acc={acc*100:.1f}%  "
              f"recovery: {rec}  reinc={st['reincarnations']}")

        acc, lc, st = run_ppr_condition(seed, tentacles=True, reincarnation=True)
        accs["ppr_full"].append(acc)
        curves["ppr_full"].append(lc)
        reinc_counts["ppr_full"].append(st["reincarnations"])
        rec = recovery_steps(lc, spike_steps)
        print(f"    ppr_full        acc={acc*100:.1f}%  "
              f"recovery: {rec}  reinc={st['reincarnations']} "
              f"probes={st['probes']} success={st['successful_pulls']}")

    print("\n  === SUMMARY ===")
    results = {}
    for cond in conditions:
        mean_acc, std_acc = stat([v * 100 for v in accs[cond]])
        results[f"{cond}_accuracy"] = dict(mean=round(mean_acc, 2), std=round(std_acc, 2))
        print(f"  {cond:18s}  acc={mean_acc:.2f}% ± {std_acc:.2f}%", end="")
        if cond in reinc_counts:
            mean_r, _ = stat(reinc_counts[cond])
            results[f"{cond}_reinc"] = round(mean_r, 1)
            print(f"  reinc/run={mean_r:.1f}", end="")
        print()

    # Average loss curves across seeds
    avg_curves = {}
    for cond in conditions:
        seed_curves = curves[cond]
        n = min(len(lc) for lc in seed_curves)
        avg = [np.nanmean([lc[i] for lc in seed_curves]) for i in range(n)]
        avg_curves[cond] = avg

    with open("exp4b_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot(avg_curves, spike_steps)
    print(f"\nDone in {time.time()-t0:.1f}s  →  exp4b_results.json + loss_spike_recovery.png")


def _plot(avg_curves, spike_steps):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"plain_momentum": "gray", "reinc_only": "tab:orange", "ppr_full": "tab:blue"}
    labels = {"plain_momentum": "Plain momentum",
              "reinc_only": "Reincarnation only",
              "ppr_full": "PPR full (coord_align + reinc)"}
    for cond, curve in avg_curves.items():
        ax.plot(curve, label=labels[cond], color=colors[cond], linewidth=1.8)
    for sp in spike_steps:
        ax.axvline(sp, color="red", linewidth=1.2, linestyle="--", alpha=0.6,
                   label="Loss spike" if sp == spike_steps[0] else None)
    ax.set_xlabel("Training step")
    ax.set_ylabel("True-label training loss")
    ax.set_title("Loss spike recovery: plain momentum vs reincarnation (digits, avg over 5 seeds)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_spike_recovery.png", dpi=130)
    print("Saved loss_spike_recovery.png")


if __name__ == "__main__":
    main()
