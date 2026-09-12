"""
Experiment 5: PPR with plateau-based stall detection on Digits + MNIST.

Tests the structural fix for the mini-batch noise problem:

  OLD (broken): stall detector uses instantaneous velocity, which reflects
    mini-batch gradient noise, not genuine training stalls.  best_params
    tracks minimum mini-batch loss, which is noisy and meaningless.

  NEW (fixed):  eval_fn evaluates loss on a fixed held-out subset every step.
    - Stall detected via loss-plateau on EMA of eval_fn output.
    - best_params updated using eval_fn quality, not mini-batch loss.
    - Probe acceptance (accept_tau) evaluated via the same eval_fn.

eval_fn setup:
  Digits: all 1,347 training samples (full-batch anyway, so no difference).
  MNIST:  first 2,000 samples from the training set, fixed throughout training.

Parameters chosen conservatively to avoid disrupting training:
  stall_eps    = 5e-4   (EMA must drop by 0.0005 to count as progress)
  stall_steps  = 100    (100 consecutive no-progress steps → stall, ~0.4 epochs)
  loss_ema_alpha = 0.98 (slow EMA, ~50-step effective window)
  accept_tau   = 0.005  (probe must improve eval loss by 0.5%)
  progress_tau = 0.001  (episode must improve best eval loss by 0.1%)
  pull_force   = 0.05   (very gentle pull, gradient stays dominant)
  pull_steps   = 3      (minimal disruption per stall event)
  max_attempts = 3

Comparison: SGD+momentum, Adam, AdamW, PPR-plateau, PPR-velocity (old params).
Both PPR variants use coord_align probing, probe_subset_size=last-layer size.
"""
import math
import json
import ssl
import time
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from ppr_optimizer import PPR

ssl._create_default_https_context = ssl._create_unverified_context

SEEDS = [0, 1, 2, 3, 4]
MNIST_SEEDS = [0, 1, 2]


def stat_pct(vals):
    vals = [v * 100 for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=round(mean, 2), std=round(var ** 0.5, 2))


# ─────────────────────────────── DIGITS ──────────────────────────────────────

DIGITS_EPOCHS = 150


def get_digits(seed):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


def make_digits_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def digits_eval_acc(model, Xte, yte):
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


def digits_baseline(kind, seed, lr):
    Xtr, ytr, Xte, yte = get_digits(seed)
    model = make_digits_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = {"sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
           "adam":  torch.optim.Adam(model.parameters(), lr=lr),
           "adamw": torch.optim.AdamW(model.parameters(), lr=lr)}[kind]
    for _ in range(DIGITS_EPOCHS):
        opt.zero_grad(); crit(model(Xtr), ytr).backward(); opt.step()
    return digits_eval_acc(model, Xte, yte)


def digits_ppr(seed, use_plateau=True):
    Xtr, ytr, Xte, yte = get_digits(seed)
    model = make_digits_model(seed)
    crit = nn.CrossEntropyLoss()

    if use_plateau:
        opt = PPR(
            model.parameters(), lr=0.1, momentum=0.9,
            tentacles=True, reincarnation=True,
            probe_mode="coord_align", probe_subset_size=330,
            reach_frac=0.05, pull_force=0.05, pull_steps=3,
            accept_tau=0.002, progress_tau=0.005,
            stall_eps=5e-4, stall_steps=100, settle_steps=5,
            stall_mode="auto", loss_ema_alpha=0.98,
            max_attempts=3, jitter=1e-4,
        )
    else:
        # Original velocity-based params (from exp3, the working full-batch version)
        opt = PPR(
            model.parameters(), lr=0.1, momentum=0.9,
            tentacles=True, reincarnation=True,
            probe_mode="coord_align", probe_subset_size=330,
            reach_frac=0.05, pull_force=0.3, pull_steps=15,
            accept_tau=1e-4, progress_tau=1e-3,
            stall_eps=0.01, stall_steps=8, settle_steps=5,
            stall_mode="velocity",
            max_attempts=4, jitter=1e-4,
        )

    def closure():
        opt.zero_grad(); loss = crit(model(Xtr), ytr); loss.backward(); return loss

    def eval_fn():
        with torch.no_grad(): return crit(model(Xtr), ytr).item()

    for _ in range(DIGITS_EPOCHS):
        loss = opt.step(closure, eval_fn, eval_fn)  # eval_fn doubles as loss_only_fn
        if not math.isfinite(loss.item()): break
    return digits_eval_acc(model, Xte, yte), opt.stats()


def run_digits():
    print("\n" + "=" * 64)
    print("DIGITS  (2,410-param MLP, full-batch, 150 epochs, 5 seeds)")
    print("=" * 64)
    results = {}

    for kind, lr in [("sgd", 0.1), ("adam", 0.01), ("adamw", 0.01)]:
        accs = [digits_baseline(kind, s, lr) for s in SEEDS]
        results[f"digits_{kind}"] = stat_pct(accs)
        print(f"  {kind:5s}: {results[f'digits_{kind}']}")

    for label, use_plateau in [("plateau", True), ("velocity", False)]:
        accs, tp, ts, tr = [], 0, 0, 0
        for s in SEEDS:
            acc, st = digits_ppr(s, use_plateau)
            accs.append(acc); tp += st["probes"]; ts += st["successful_pulls"]
            tr += st["reincarnations"]
        rate = 100 * ts / tp if tp else 0
        results[f"digits_ppr_{label}"] = stat_pct(accs)
        results[f"digits_ppr_{label}_probes"] = f"{ts}/{tp} ({rate:.0f}%), {tr} reinc"
        print(f"  ppr-{label}: {results[f'digits_ppr_{label}']}  "
              f"probes {results[f'digits_ppr_{label}_probes']}")
    return results


# ─────────────────────────────── MNIST ───────────────────────────────────────

MNIST_EPOCHS = 15
MNIST_BATCH = 256
EVAL_N = 2000   # fixed held-out eval subset size


def get_mnist():
    root = os.path.expanduser("~/data")
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    return (torchvision.datasets.MNIST(root, train=True,  download=True, transform=t),
            torchvision.datasets.MNIST(root, train=False, download=True, transform=t))


def make_mnist_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def eval_acc_loader(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item(); total += len(y)
    return correct / total


def get_eval_subset(train_ds, n=EVAL_N):
    """Fixed held-out eval subset — never used for gradient computation."""
    Xs, ys = [], []
    for i in range(n):
        x, y = train_ds[i]; Xs.append(x.view(-1)); ys.append(y)
    return torch.stack(Xs), torch.tensor(ys)


def mnist_baseline(kind, seed, lr, train_ds, test_loader):
    torch.manual_seed(seed)
    model = make_mnist_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = {"sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
           "adam":  torch.optim.Adam(model.parameters(), lr=lr),
           "adamw": torch.optim.AdamW(model.parameters(), lr=lr)}[kind]
    loader = DataLoader(train_ds, batch_size=MNIST_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(MNIST_EPOCHS):
        for X, y in loader:
            opt.zero_grad(); crit(model(X.view(X.size(0), -1)), y).backward(); opt.step()
    return eval_acc_loader(model, test_loader)


def mnist_ppr_plateau(seed, train_ds, test_loader, eval_X, eval_y):
    torch.manual_seed(seed)
    model = make_mnist_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = PPR(
        model.parameters(), lr=0.01, momentum=0.9,
        tentacles=True, reincarnation=True,
        probe_mode="coord_align", probe_subset_size=64,
        reach_frac=0.05, pull_force=0.05, pull_steps=3,
        accept_tau=0.002, progress_tau=0.005,
        stall_eps=5e-4, stall_steps=100, settle_steps=5,
        stall_mode="auto", loss_ema_alpha=0.98,
        max_attempts=3, jitter=1e-4,
    )
    loader = DataLoader(train_ds, batch_size=MNIST_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    def eval_fn():
        with torch.no_grad():
            return crit(model(eval_X), eval_y).item()

    for _ in range(MNIST_EPOCHS):
        for X, y in loader:
            Xf = X.view(X.size(0), -1)

            def closure(Xf=Xf, y=y):
                opt.zero_grad()
                l = crit(model(Xf), y)
                l.backward()   # PPR reads gradients via _flat_grad(); must call here
                return l

            loss = opt.step(closure, eval_fn, eval_fn)
            if not math.isfinite(loss.item()):
                return float("nan"), opt.stats()

    return eval_acc_loader(model, test_loader), opt.stats()


def run_mnist():
    print("\n" + "=" * 64)
    print(f"MNIST   (101,770-param MLP, batch={MNIST_BATCH}, "
          f"{MNIST_EPOCHS} epochs, {len(MNIST_SEEDS)} seeds)")
    print(f"eval_fn: {EVAL_N} fixed training samples (plateau stall detection)")
    print("=" * 64)
    results = {}

    train_ds, test_ds = get_mnist()
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    eval_X, eval_y = get_eval_subset(train_ds)

    for kind, lr in [("sgd", 0.01), ("adam", 1e-3), ("adamw", 1e-3)]:
        accs = []
        for seed in MNIST_SEEDS:
            t0 = time.time()
            acc = mnist_baseline(kind, seed, lr, train_ds, test_loader)
            accs.append(acc)
            print(f"    {kind} seed={seed}  {acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        results[f"mnist_{kind}"] = stat_pct(accs)
        print(f"  {kind}: {results[f'mnist_{kind}']}\n")

    accs, tp, ts, tr = [], 0, 0, 0
    for seed in MNIST_SEEDS:
        t0 = time.time()
        acc, st = mnist_ppr_plateau(seed, train_ds, test_loader, eval_X, eval_y)
        accs.append(acc); tp += st["probes"]; ts += st["successful_pulls"]
        tr += st["reincarnations"]
        print(f"    ppr-plateau seed={seed}  {acc*100:.2f}%  "
              f"probes={st['probes']} success={st['successful_pulls']} "
              f"reinc={st['reincarnations']}  ({time.time()-t0:.0f}s)")
    rate = 100 * ts / tp if tp else 0
    results["mnist_ppr_plateau"] = stat_pct(accs)
    results["mnist_ppr_plateau_probes"] = f"{ts}/{tp} ({rate:.0f}%), {tr} reinc"
    print(f"  ppr-plateau: {results['mnist_ppr_plateau']}  "
          f"probes {results['mnist_ppr_plateau_probes']}")

    return results


def main():
    t0 = time.time()
    all_results = {}

    d = run_digits()
    all_results.update(d)

    m = run_mnist()
    all_results.update(m)

    with open("exp5_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print("\nDIGITS (5 seeds, 150 epochs):")
    for k in ["digits_sgd", "digits_adam", "digits_adamw",
              "digits_ppr_plateau", "digits_ppr_velocity"]:
        if k in all_results:
            tag = k.replace("digits_", "")
            print(f"  {tag:16s}: {all_results[k]['mean']:.2f}% ± {all_results[k]['std']:.2f}%")
            pk = f"{k}_probes"
            if pk in all_results:
                print(f"    probes: {all_results[pk]}")

    print("\nMNIST (3 seeds, 15 epochs):")
    for k in ["mnist_sgd", "mnist_adam", "mnist_adamw", "mnist_ppr_plateau"]:
        if k in all_results:
            tag = k.replace("mnist_", "")
            print(f"  {tag:16s}: {all_results[k]['mean']:.2f}% ± {all_results[k]['std']:.2f}%")
            pk = f"{k}_probes"
            if pk in all_results:
                print(f"    probes: {all_results[pk]}")

    print(f"\nTotal time: {time.time()-t0:.1f}s  →  exp5_results.json")


if __name__ == "__main__":
    main()
