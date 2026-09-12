"""
Experiment 3: Digits (sklearn) + MNIST with coord_align probing.

Two datasets, same comparison structure:
  Baselines: SGD+momentum, Adam, AdamW
  PPR: coord_align probing (stall_eps=0.01, tuned from calibration in exp1c)

DIGITS (sklearn load_digits)
  Full-batch. Model: Linear(64,32)->ReLU->Linear(32,10), 2,410 params.
  Last layer = 330 params → probe_subset_size=330.
  Two conditions:
    Normal: SGD lr=0.1, Adam/AdamW lr=0.01, PPR lr=0.1
    Stress:  SGD lr=2.0, Adam lr=0.5, AdamW lr=0.5, PPR lr=2.0 + 20% label noise
  150 epochs, 5 seeds.

MNIST (torchvision)
  Mini-batch (batch_size=256). Model: Linear(784,128)->ReLU->Linear(128,10), ~101K params.
  Last layer = 1,290 params → probe_subset_size=64 (cheap subset of output layer).
  Normal condition only (stress omitted on CPU — 30 epochs too expensive with high-LR chaos).
  20 epochs, 3 seeds.

Stall detection calibration note:
  At equilibrium, RMS(velocity) ≈ lr * RMS(grad) / (1-momentum).
  For digits at lr=0.1, well-converged CrossEntropy RMS(grad) ≈ 0.05-0.1,
  so equilibrium RMS(v) ≈ 0.05-0.1. stall_eps=0.01 fires only when velocity
  has decayed substantially below equilibrium — a genuine stall.
  stall_eps=0.05 fires during productive training (confirmed in pilot run → high variance).
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
from ppr_optimizer import PPR

# SSL fix for macOS certificate issue when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SIM905

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

SEEDS = [0, 1, 2, 3, 4]


def stat_pct(vals):
    vals = [v * 100 for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=round(mean, 2), std=round(var ** 0.5, 2))


# ─────────────────────────────── DIGITS ──────────────────────────────────────

DIGITS_EPOCHS = 150


def get_digits(seed, label_noise=0.0):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    if label_noise > 0:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ytr), int(label_noise * len(ytr)), replace=False)
        ytr = ytr.copy()
        ytr[idx] = rng.randint(0, 10, size=len(idx))
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


def make_digits_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def eval_acc(model, Xte, yte):
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


def digits_baseline(kind, seed, lr, label_noise=0.0):
    Xtr, ytr, Xte, yte = get_digits(seed, label_noise)
    model = make_digits_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = {"sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
           "adam":  torch.optim.Adam(model.parameters(), lr=lr),
           "adamw": torch.optim.AdamW(model.parameters(), lr=lr)}[kind]
    for _ in range(DIGITS_EPOCHS):
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        if not math.isfinite(loss.item()):
            break
        loss.backward()
        opt.step()
    return eval_acc(model, Xte, yte)


def digits_ppr(seed, lr, stall_eps=0.01, label_noise=0.0):
    Xtr, ytr, Xte, yte = get_digits(seed, label_noise)
    model = make_digits_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = PPR(
        model.parameters(), lr=lr, momentum=0.9,
        tentacles=True, reincarnation=True,
        probe_mode="coord_align", probe_subset_size=330,   # full last layer
        reach_frac=0.05, pull_force=0.3, pull_steps=15,
        accept_tau=1e-4, progress_tau=1e-3,
        stall_eps=stall_eps, stall_steps=8, settle_steps=5,
        max_attempts=4, jitter=1e-4,
    )

    def closure():
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward()
        return loss

    def loss_only():
        with torch.no_grad():
            return crit(model(Xtr), ytr).item()

    for _ in range(DIGITS_EPOCHS):
        loss = opt.step(closure, loss_only)
        if not math.isfinite(loss.item()):
            break
    return eval_acc(model, Xte, yte), opt.stats()


def run_digits():
    print("\n" + "=" * 62)
    print("DIGITS  (sklearn, 2,410-param MLP, full-batch, 150 epochs)")
    print("=" * 62)
    results = {}

    # ── Normal condition ──
    print("\n--- normal (reasonable LR, no noise) ---")
    for kind, lr in [("sgd", 0.1), ("adam", 0.01), ("adamw", 0.01)]:
        accs = [digits_baseline(kind, s, lr) for s in SEEDS]
        results[f"digits_{kind}_normal"] = stat_pct(accs)
        print(f"  {kind:6s} lr={lr}  : {results[f'digits_{kind}_normal']}")

    accs_n, p_n, s_n = [], 0, 0
    for seed in SEEDS:
        acc, st = digits_ppr(seed, lr=0.1, stall_eps=0.01)
        accs_n.append(acc)
        p_n += st["probes"]; s_n += st["successful_pulls"]
    results["digits_ppr_normal"] = stat_pct(accs_n)
    rate_n = 100 * s_n / p_n if p_n else 0
    results["digits_ppr_normal_probes"] = f"{s_n}/{p_n} ({rate_n:.0f}% success)"
    print(f"  ppr    lr=0.1 : {results['digits_ppr_normal']}  "
          f"probes {results['digits_ppr_normal_probes']}")

    # ── Stress condition: high LR + 20% label noise, same LR for PPR ──
    print("\n--- stress (high LR + 20% label noise) ---")
    stress_lrs = {"sgd": 2.0, "adam": 0.5, "adamw": 0.5, "ppr": 2.0}
    for kind, lr in [("sgd", stress_lrs["sgd"]),
                     ("adam", stress_lrs["adam"]),
                     ("adamw", stress_lrs["adamw"])]:
        accs = [digits_baseline(kind, s, lr, label_noise=0.2) for s in SEEDS]
        results[f"digits_{kind}_stress"] = stat_pct(accs)
        print(f"  {kind:6s} lr={lr}  : {results[f'digits_{kind}_stress']}")

    accs_s, p_s, s_s = [], 0, 0
    for seed in SEEDS:
        acc, st = digits_ppr(seed, lr=stress_lrs["ppr"], stall_eps=0.01, label_noise=0.2)
        accs_s.append(acc)
        p_s += st["probes"]; s_s += st["successful_pulls"]
    results["digits_ppr_stress"] = stat_pct(accs_s)
    rate_s = 100 * s_s / p_s if p_s else 0
    results["digits_ppr_stress_probes"] = f"{s_s}/{p_s} ({rate_s:.0f}% success)"
    print(f"  ppr    lr={stress_lrs['ppr']}   : {results['digits_ppr_stress']}  "
          f"probes {results['digits_ppr_stress_probes']}")

    return results


# ─────────────────────────────── MNIST ───────────────────────────────────────

MNIST_EPOCHS = 20
MNIST_BATCH = 256
MNIST_SEEDS = [0, 1, 2]   # 3 seeds on CPU keeps runtime reasonable


def get_mnist():
    root = os.path.expanduser("~/data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(root, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
    return train_ds, test_ds


def make_mnist_model(seed):
    torch.manual_seed(seed)
    # 784 → 128 → 10  (101,770 params; last layer = 1,290)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def eval_mnist(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


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
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()
    return eval_mnist(model, test_loader)


def mnist_ppr(seed, stall_eps, train_ds, test_loader):
    torch.manual_seed(seed)
    model = make_mnist_model(seed)
    crit = nn.CrossEntropyLoss()
    # probe_subset_size=64: last 64 params of flat vector (tail of output layer).
    # coord_align → 128 forward passes per stall event on current mini-batch.
    opt = PPR(
        model.parameters(), lr=0.01, momentum=0.9,
        tentacles=True, reincarnation=True,
        probe_mode="coord_align", probe_subset_size=64,
        reach_frac=0.05, pull_force=0.3, pull_steps=10,
        accept_tau=1e-4, progress_tau=1e-3,
        stall_eps=stall_eps, stall_steps=8, settle_steps=5,
        max_attempts=4, jitter=1e-4,
    )
    loader = DataLoader(train_ds, batch_size=MNIST_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(MNIST_EPOCHS):
        for X, y in loader:
            X_b, y_b = X, y

            def closure():
                opt.zero_grad()
                return crit(model(X_b), y_b)

            def loss_only():
                with torch.no_grad():
                    return crit(model(X_b), y_b).item()

            loss = opt.step(closure, loss_only)
            if not math.isfinite(loss.item()):
                return float("nan"), opt.stats()
    return eval_mnist(model, test_loader), opt.stats()


def run_mnist():
    print("\n" + "=" * 62)
    print(f"MNIST   (torchvision, 101,770-param MLP, batch={MNIST_BATCH}, "
          f"{MNIST_EPOCHS} epochs, {len(MNIST_SEEDS)} seeds)")
    print("=" * 62)
    results = {}

    print("  Downloading MNIST...")
    try:
        train_ds, test_ds = get_mnist()
        test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    except Exception as e:
        print(f"  MNIST download failed: {e}")
        print("  Skipping MNIST section.")
        return {"mnist_error": str(e)}

    print("  Done.  Training baselines...")
    for kind, lr in [("sgd", 0.01), ("adam", 1e-3), ("adamw", 1e-3)]:
        accs = []
        for seed in MNIST_SEEDS:
            t0 = time.time()
            acc = mnist_baseline(kind, seed, lr, train_ds, test_loader)
            accs.append(acc)
            print(f"    {kind} seed={seed}  acc={acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        results[f"mnist_{kind}"] = stat_pct(accs)
        print(f"  {kind:6s} lr={lr}  : {results[f'mnist_{kind}']}")

    for stall_eps, label in [(0.01, "mnist_ppr_eps001"), (0.05, "mnist_ppr_eps005")]:
        accs, total_p, total_s = [], 0, 0
        for seed in MNIST_SEEDS:
            t0 = time.time()
            acc, st = mnist_ppr(seed, stall_eps, train_ds, test_loader)
            accs.append(acc)
            total_p += st["probes"]; total_s += st["successful_pulls"]
            print(f"    ppr eps={stall_eps} seed={seed}  acc={acc*100:.2f}%  "
                  f"probes={st['probes']} success={st['successful_pulls']}  ({time.time()-t0:.0f}s)")
        rate = 100 * total_s / total_p if total_p else 0
        results[label] = stat_pct(accs)
        results[f"{label}_probes"] = f"{total_s}/{total_p} ({rate:.0f}% success)"
        print(f"  ppr eps={stall_eps}: {results[label]}  probes {results[f'{label}_probes']}")

    return results


def main():
    t0 = time.time()
    all_results = {}

    digits_res = run_digits()
    all_results.update(digits_res)

    mnist_res = run_mnist()
    all_results.update(mnist_res)

    with open("exp3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Results saved to exp3_results.json")

    _print_summary(all_results)


def _print_summary(r):
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print("\nDIGITS — normal (5 seeds, 150 epochs):")
    for k in ["digits_sgd_normal", "digits_adam_normal", "digits_adamw_normal",
              "digits_ppr_normal"]:
        if k in r:
            tag = k.replace("digits_", "").replace("_normal", "")
            print(f"  {tag:10s}: {r[k]['mean']:.2f}% ± {r[k]['std']:.2f}%")
    if "digits_ppr_normal_probes" in r:
        print(f"  PPR probes: {r['digits_ppr_normal_probes']}")

    print("\nDIGITS — stress (high LR + 20% label noise, 5 seeds):")
    for k in ["digits_sgd_stress", "digits_adam_stress", "digits_adamw_stress",
              "digits_ppr_stress"]:
        if k in r:
            tag = k.replace("digits_", "").replace("_stress", "")
            print(f"  {tag:10s}: {r[k]['mean']:.2f}% ± {r[k]['std']:.2f}%")
    if "digits_ppr_stress_probes" in r:
        print(f"  PPR probes: {r['digits_ppr_stress_probes']}")

    print("\nMNIST — normal (3 seeds, 20 epochs):")
    for k in ["mnist_sgd", "mnist_adam", "mnist_adamw",
              "mnist_ppr_eps001", "mnist_ppr_eps005"]:
        if k in r:
            tag = k.replace("mnist_", "")
            print(f"  {tag:20s}: {r[k]['mean']:.2f}% ± {r[k]['std']:.2f}%")
            probe_key = f"{k}_probes"
            if probe_key in r:
                print(f"    probes: {r[probe_key]}")


if __name__ == "__main__":
    main()
