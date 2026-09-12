"""
Experiment 6: Strong loss spikes on MNIST — does reincarnation help?

Prior spike test (exp4b) failed because 1-step random-label spikes at lr=0.1
were too weak: plain momentum recovered within a single training step, so
reincarnation never had a window to act.

This experiment uses spikes large enough that recovery is genuinely slow:
  - 10 consecutive random-label gradient steps at lr=0.05 (5× normal)
  - Applied 3 times during training (at ~epoch 4, 8, 12 of 20)
  - MNIST, 101,770-param MLP — a real network, not a toy function

Conditions (3 seeds each):
  no_spike         : clean SGD baseline (reference ceiling)
  plain_momentum   : SGD with spikes (reference floor — how bad is the damage?)
  reinc_only       : PPR reincarnation + plateau stall detection, with spikes
  ppr_full         : coord_align probing + reincarnation, with spikes

PPR uses eval_fn (2000-sample fixed subset) for plateau stall detection and
best_params tracking — the structural fix from exp5. Closure calls backward().

Metrics:
  - Test accuracy after every epoch
  - eval_fn loss every 100 steps (fine-grained view of recovery)
  - Reincarnation events (step number + quality at that point)
  - Probe events for ppr_full
  - Recovery speed: steps from spike to return within 1% of pre-spike accuracy
"""
import math, json, ssl, os, time
import torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ppr_optimizer import PPR

ssl._create_default_https_context = ssl._create_unverified_context

SEEDS = [0, 1, 2]
EPOCHS = 20
BATCH = 256
EVAL_N = 2000
SPIKE_EVERY_STEPS = 1000   # spike at step 1000, 2000, 3000
SPIKE_N_STEPS = 10          # 10 consecutive random-label gradient steps per spike
SPIKE_LR = 0.05             # 5× the normal training lr of 0.01
LOG_EVERY = 100             # log eval_fn loss every N steps


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def get_mnist():
    root = os.path.expanduser("~/data")
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    return (torchvision.datasets.MNIST(root, train=True,  download=True, transform=t),
            torchvision.datasets.MNIST(root, train=False, download=True, transform=t))


def get_eval_subset(train_ds, n=EVAL_N):
    Xs, ys = [], []
    for i in range(n):
        x, y = train_ds[i]; Xs.append(x.view(-1)); ys.append(y)
    return torch.stack(Xs), torch.tensor(ys)


def test_acc(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item(); total += len(y)
    return correct / total


def apply_spike(model, spike_steps=SPIKE_N_STEPS, spike_lr=SPIKE_LR):
    """
    Inject N high-LR random-label gradient steps directly.
    Uses a fresh zero-momentum SGD — does NOT go through PPR's closure
    so PPR's velocity state is NOT corrupted by the spike gradient.
    This is deliberate: PPR's pre-spike momentum points toward the
    true minimum, which helps recovery.  The spike only moves the
    parameters, not the momentum.
    """
    crit_local = nn.CrossEntropyLoss()
    tmp = torch.optim.SGD(model.parameters(), lr=spike_lr, momentum=0.0)
    with torch.enable_grad():
        for _ in range(spike_steps):
            tmp.zero_grad()
            X_r = torch.randn(BATCH, 784)
            y_r = torch.randint(0, 10, (BATCH,))
            crit_local(model(X_r), y_r).backward()
            tmp.step()


def run_condition(kind, seed, train_ds, test_loader, eval_X, eval_y, with_spikes):
    torch.manual_seed(seed)
    model = make_model(seed)
    crit = nn.CrossEntropyLoss()

    if kind == "plain_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        use_ppr = False
    elif kind in ("reinc_only", "ppr_full"):
        use_ppr = True
        opt = PPR(
            model.parameters(), lr=0.01, momentum=0.9,
            tentacles=(kind == "ppr_full"),
            reincarnation=True,
            probe_mode="coord_align", probe_subset_size=64,
            reach_frac=0.05, pull_force=0.05, pull_steps=5,
            accept_tau=0.002, progress_tau=0.005,
            stall_eps=5e-4, stall_steps=50,   # shorter than exp5 to detect post-spike stalls faster
            settle_steps=5, loss_ema_alpha=0.98,
            stall_mode="auto",
            max_attempts=3, jitter=1e-3,       # larger jitter than before for more exploration
        )
    else:
        raise ValueError(kind)

    def eval_fn():
        with torch.no_grad():
            return crit(model(eval_X), eval_y).item()

    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    step = 0
    spike_steps_fired = []
    epoch_accs = []
    loss_log = []          # (step, eval_loss)
    reinc_log = []         # (step, eval_loss) at each reincarnation
    pre_spike_accs = {}    # step → test acc just before spike

    for epoch in range(EPOCHS):
        for X, y in loader:
            # ── spike injection ──────────────────────────────────────
            if with_spikes and step > 0 and step % SPIKE_EVERY_STEPS == 0:
                pre_spike_accs[step] = test_acc(model, test_loader)
                apply_spike(model)
                spike_steps_fired.append(step)
                print(f"      spike at step {step}  "
                      f"pre-acc={pre_spike_accs[step]*100:.1f}%  "
                      f"post-spike eval={eval_fn():.4f}")

            # ── training step ────────────────────────────────────────
            Xf = X.view(X.size(0), -1)
            if use_ppr:
                prev_reinc = opt.stats()["reincarnations"]

                def closure(Xf=Xf, y=y):
                    opt.zero_grad()
                    l = crit(model(Xf), y)
                    l.backward()
                    return l

                opt.step(closure, eval_fn, eval_fn)

                new_reinc = opt.stats()["reincarnations"]
                if new_reinc > prev_reinc:
                    reinc_log.append((step, eval_fn()))
            else:
                opt.zero_grad()
                crit(model(Xf), y).backward()
                opt.step()

            # ── logging ──────────────────────────────────────────────
            if step % LOG_EVERY == 0:
                loss_log.append((step, eval_fn()))

            step += 1

        epoch_accs.append(test_acc(model, test_loader))

    ppr_stats = opt.stats() if use_ppr else {}
    return dict(
        epoch_accs=epoch_accs,
        loss_log=loss_log,
        spike_steps=spike_steps_fired,
        pre_spike_accs=pre_spike_accs,
        reinc_log=reinc_log,
        ppr_stats=ppr_stats,
    )


def recovery_analysis(results_by_seed, test_loader, train_ds):
    """For each spike, measure steps until accuracy returns to within 1% of pre-spike."""
    all_recoveries = []
    for r in results_by_seed:
        for sp, pre_acc in r["pre_spike_accs"].items():
            target = pre_acc - 0.01  # within 1 percentage point
            # find first epoch_acc after the spike that exceeds target
            steps_per_epoch = 234   # MNIST 60K / 256
            spike_epoch = sp // steps_per_epoch
            recovered = None
            for epoch_idx, acc in enumerate(r["epoch_accs"]):
                if epoch_idx >= spike_epoch and acc >= target:
                    recovered = (epoch_idx - spike_epoch) * steps_per_epoch
                    break
            all_recoveries.append(recovered)
    finite = [x for x in all_recoveries if x is not None]
    return (sum(finite)/len(finite) if finite else float("inf"),
            sum(1 for x in all_recoveries if x is None))


def stat_pct(vals):
    vals = [v*100 for v in vals if math.isfinite(v)]
    if not vals: return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals)/len(vals)
    var = sum((v-mean)**2 for v in vals)/max(1, len(vals)-1)
    return dict(mean=round(mean,2), std=round(var**0.5,2))


def main():
    t0 = time.time()
    train_ds, test_ds = get_mnist()
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    eval_X, eval_y = get_eval_subset(train_ds)

    print(f"MNIST strong-spike experiment")
    print(f"Spike: {SPIKE_N_STEPS} steps × lr={SPIKE_LR} random-label, "
          f"every {SPIKE_EVERY_STEPS} steps, 3 spikes total")
    print(f"Network: 784→128→ReLU→10  (101,770 params)")
    print(f"Seeds: {SEEDS}  Epochs: {EPOCHS}\n")

    conditions = ["no_spike", "plain_momentum", "reinc_only", "ppr_full"]
    all_results = {c: [] for c in conditions}

    for cond in conditions:
        with_spikes = (cond != "no_spike")
        kind = "plain_momentum" if cond == "no_spike" else cond
        print(f"  ── {cond} {'(no spikes)' if not with_spikes else '(WITH spikes)'} ──")
        for seed in SEEDS:
            t1 = time.time()
            r = run_condition(kind, seed, train_ds, test_loader,
                              eval_X, eval_y, with_spikes)
            all_results[cond].append(r)
            st = r["ppr_stats"]
            probes = st.get("probes", 0); success = st.get("successful_pulls", 0)
            reinc = st.get("reincarnations", 0)
            final_acc = r["epoch_accs"][-1]*100
            print(f"    seed={seed}  final={final_acc:.2f}%  "
                  f"probes={probes} success={success} reinc={reinc}  "
                  f"({time.time()-t1:.0f}s)")
        final_accs = [r["epoch_accs"][-1] for r in all_results[cond]]
        print(f"  {cond}: {stat_pct(final_accs)}\n")

    # ── summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond in conditions:
        final_accs = [r["epoch_accs"][-1] for r in all_results[cond]]
        s = stat_pct(final_accs)
        label = "(no spikes)" if cond == "no_spike" else "(WITH spikes)"
        avg_rec, n_norecover = recovery_analysis(
            all_results[cond], test_loader, train_ds)
        rec_str = (f"avg recovery ~{avg_rec:.0f} steps" if math.isfinite(avg_rec)
                   else "N/A")
        reinc_total = sum(len(r["reinc_log"]) for r in all_results[cond])
        print(f"  {cond:18s} {label:14s}: {s['mean']:.2f}% ± {s['std']:.2f}%  "
              f"{rec_str}  reinc={reinc_total}")

    # ── save ─────────────────────────────────────────────────────────
    save = {}
    for cond, runs in all_results.items():
        save[cond] = [{
            "epoch_accs": r["epoch_accs"],
            "loss_log": r["loss_log"],
            "spike_steps": r["spike_steps"],
            "reinc_log": r["reinc_log"],
            "ppr_stats": {k: v for k, v in r["ppr_stats"].items()
                          if not isinstance(v, torch.Tensor)},
        } for r in runs]
    with open("exp6_results.json", "w") as f:
        json.dump(save, f, indent=2)

    _plot(all_results)
    print(f"\nTotal time: {time.time()-t0:.1f}s  →  exp6_results.json + spike_recovery.png")


def _plot(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: test accuracy by epoch, averaged across seeds
    colors = {"no_spike": "black", "plain_momentum": "gray",
              "reinc_only": "tab:orange", "ppr_full": "tab:blue"}
    labels = {"no_spike": "No spikes (ceiling)",
              "plain_momentum": "Plain momentum + spikes",
              "reinc_only": "Reincarnation only + spikes",
              "ppr_full": "PPR full + spikes"}
    ax = axes[0]
    for cond, runs in all_results.items():
        n_epochs = len(runs[0]["epoch_accs"])
        avg = [np.mean([r["epoch_accs"][e]*100 for r in runs]) for e in range(n_epochs)]
        ax.plot(range(1, n_epochs+1), avg, label=labels[cond],
                color=colors[cond], linewidth=2 if "ppr" in cond else 1.4)
    # Mark spike epochs
    spike_epochs = [SPIKE_EVERY_STEPS // 234 + 1,
                    2*SPIKE_EVERY_STEPS // 234 + 1,
                    3*SPIKE_EVERY_STEPS // 234 + 1]
    for se in spike_epochs:
        if se <= EPOCHS:
            ax.axvline(se, color="red", linestyle="--", linewidth=1, alpha=0.6,
                       label="Spike" if se == spike_epochs[0] else None)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy over training\n(spikes marked in red)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: eval_fn loss log (fine-grained)
    ax2 = axes[1]
    for cond, runs in all_results.items():
        if cond == "no_spike": continue
        all_steps = sorted(set(s for r in runs for s,_ in r["loss_log"]))
        avg_loss = []
        for step in all_steps:
            vals = [dict(r["loss_log"]).get(step) for r in runs
                    if step in dict(r["loss_log"])]
            vals = [v for v in vals if v is not None]
            if vals: avg_loss.append((step, np.mean(vals)))
        if avg_loss:
            xs, ys = zip(*avg_loss)
            ax2.plot(xs, ys, label=labels[cond], color=colors[cond], linewidth=1.5)
    for sp in [SPIKE_EVERY_STEPS, 2*SPIKE_EVERY_STEPS, 3*SPIKE_EVERY_STEPS]:
        ax2.axvline(sp, color="red", linestyle="--", linewidth=1, alpha=0.6,
                    label="Spike" if sp == SPIKE_EVERY_STEPS else None)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Eval loss (2K fixed samples)")
    ax2.set_title("Eval loss (fine-grained)\npost-spike recovery visible here")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spike_recovery.png", dpi=130)
    print("Saved spike_recovery.png")


if __name__ == "__main__":
    main()
