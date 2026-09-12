"""
Experiment 4a: Tuned PPR on MNIST.

Tests whether the MNIST catastrophe (10.76% accuracy) from exp3 was a
tuning problem or a structural one. Three specific changes from exp3:

  1. stall_steps 8 → 200    -- prevents stall detector from firing on
                               mini-batch gradient noise every 23 steps
  2. accept_tau 1e-4 → 0.01 -- requires a real improvement on the probe
                               batch, not noise-level acceptance
  3. pull_steps 10 → 5      -- halves disruption per stall event
  4. pull_force 0.3 → 0.1   -- softer pull so gradient still has influence
  5. probe on fixed held-out batch of 1000 samples (not current mini-batch)
                            -- removes the mini-batch false-positive problem

Prediction: if the MNIST failure was purely a tuning problem, PPR should
reach baseline-comparable accuracy (~97-98%) with these changes.
If it still fails or falls behind, there is a structural issue beyond tuning.

Same model and baselines as exp3: Linear(784,128)->ReLU->Linear(128,10),
batch=256, 15 epochs (enough for 97%+ on this model), 3 seeds.
"""
import math
import json
import ssl
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from ppr_optimizer import PPR

ssl._create_default_https_context = ssl._create_unverified_context

SEEDS = [0, 1, 2]
EPOCHS = 15
BATCH = 256
PROBE_N = 1000   # fixed held-out probe subset size


def stat_pct(vals):
    vals = [v * 100 for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=round(mean, 2), std=round(var ** 0.5, 2))


def get_mnist():
    root = os.path.expanduser("~/data")
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(root, train=True,  download=True, transform=t)
    test  = torchvision.datasets.MNIST(root, train=False, download=True, transform=t)
    return train, test


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def eval_acc(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def get_probe_batch(train_ds):
    """Fixed 1000-sample probe subset — stable across all steps."""
    Xs, ys = [], []
    for i in range(PROBE_N):
        x, y = train_ds[i]
        Xs.append(x.view(-1))
        ys.append(y)
    return torch.stack(Xs), torch.tensor(ys)


def run_baseline(kind, seed, lr, train_ds, test_loader):
    torch.manual_seed(seed)
    model = make_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = {"sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
           "adam":  torch.optim.Adam(model.parameters(), lr=lr),
           "adamw": torch.optim.AdamW(model.parameters(), lr=lr)}[kind]
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(EPOCHS):
        for X, y in loader:
            opt.zero_grad()
            crit(model(X.view(X.size(0), -1)), y).backward()
            opt.step()
    return eval_acc(model, test_loader)


def run_ppr(seed, train_ds, test_loader, probe_X, probe_y):
    torch.manual_seed(seed)
    model = make_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = PPR(
        model.parameters(), lr=0.01, momentum=0.9,
        tentacles=True, reincarnation=True,
        probe_mode="coord_align", probe_subset_size=64,
        reach_frac=0.05,
        pull_force=0.1,    # was 0.3 — softer pull
        pull_steps=5,      # was 10 — shorter pull
        accept_tau=0.01,   # was 1e-4 — require real improvement on probe batch
        progress_tau=0.05,
        stall_eps=0.01,
        stall_steps=200,   # was 8 — prevents noise-triggered stalls
        settle_steps=5,
        max_attempts=4, jitter=1e-4,
    )
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    for _ in range(EPOCHS):
        for X, y in loader:
            Xf = X.view(X.size(0), -1)

            def closure(Xf=Xf, y=y):
                opt.zero_grad()
                return crit(model(Xf), y)

            def loss_only(probe_X=probe_X, probe_y=probe_y):
                with torch.no_grad():
                    return crit(model(probe_X), probe_y).item()

            loss = opt.step(closure, loss_only)
            if not math.isfinite(loss.item()):
                return float("nan"), opt.stats()

    return eval_acc(model, test_loader), opt.stats()


def main():
    t0 = time.time()
    print("=" * 64)
    print(f"MNIST TUNED PPR  ({EPOCHS} epochs, batch={BATCH}, {len(SEEDS)} seeds)")
    print(f"probe batch: {PROBE_N} fixed samples (not current mini-batch)")
    print(f"stall_steps=200, accept_tau=0.01, pull_steps=5, pull_force=0.1")
    print("=" * 64)

    train_ds, test_ds = get_mnist()
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_X, probe_y = get_probe_batch(train_ds)

    results = {}
    for kind, lr in [("sgd", 0.01), ("adam", 1e-3), ("adamw", 1e-3)]:
        accs = []
        for seed in SEEDS:
            t1 = time.time()
            acc = run_baseline(kind, seed, lr, train_ds, test_loader)
            accs.append(acc)
            print(f"  {kind} seed={seed}  {acc*100:.2f}%  ({time.time()-t1:.0f}s)")
        results[f"mnist_{kind}"] = stat_pct(accs)
        print(f"  {kind}: {results[f'mnist_{kind}']}\n")

    ppr_accs, total_p, total_s, total_r = [], 0, 0, 0
    for seed in SEEDS:
        t1 = time.time()
        acc, st = run_ppr(seed, train_ds, test_loader, probe_X, probe_y)
        ppr_accs.append(acc)
        total_p += st["probes"]
        total_s += st["successful_pulls"]
        total_r += st["reincarnations"]
        print(f"  ppr seed={seed}  {acc*100:.2f}%  "
              f"probes={st['probes']} success={st['successful_pulls']} "
              f"reinc={st['reincarnations']}  ({time.time()-t1:.0f}s)")
    rate = 100 * total_s / total_p if total_p else 0
    results["mnist_ppr_tuned"] = stat_pct(ppr_accs)
    results["mnist_ppr_probe_activity"] = (
        f"{total_s}/{total_p} ({rate:.0f}% success), {total_r} reincarnations"
    )
    print(f"  ppr tuned: {results['mnist_ppr_tuned']}")
    print(f"  probe activity: {results['mnist_ppr_probe_activity']}")

    with open("exp4a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s  →  exp4a_results.json")


if __name__ == "__main__":
    main()
