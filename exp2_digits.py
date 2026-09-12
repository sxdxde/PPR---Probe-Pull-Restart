"""
Experiment 2: real dataset, small neural network.

sklearn's `load_digits` (1797 8x8 grayscale digit images, 10 classes,
no download required -- bundled with scikit-learn). A small two-layer
MLP has on the order of a few thousand parameters, which -- given what
Experiment 1 found about probe success rate collapsing between 5 and 20
dimensions -- is expected to be far too high-dimensional for PPR's
current (per-parameter random direction) probing to find anything.
This experiment exists to check that expectation directly, honestly,
rather than assume it.

Two conditions per optimizer:
  (a) normal: a reasonable learning rate for that optimizer family
  (b) stress: a deliberately too-high learning rate, plus 20% label
      noise injected into the training set, to test the robustness
      claim (Research Question 2) rather than raw accuracy.
"""
import math
import json
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ppr_optimizer import PPR

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60


def get_data(seed, label_noise=0.0):
    data = load_digits()
    X, y = data.data.astype(np.float32) / 16.0, data.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    if label_noise > 0:
        rng = np.random.RandomState(seed)
        n_flip = int(label_noise * len(ytr))
        idx = rng.choice(len(ytr), n_flip, replace=False)
        ytr = ytr.copy()
        ytr[idx] = rng.randint(0, 10, size=n_flip)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


def make_model(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def evaluate(model, Xte, yte):
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1)
        return (pred == yte).float().mean().item()


def train_baseline(kind, seed, lr, label_noise=0.0, epochs=EPOCHS):
    Xtr, ytr, Xte, yte = get_data(seed, label_noise)
    model, n_params = make_model(seed)
    crit = nn.CrossEntropyLoss()
    if kind == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif kind == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        if not math.isfinite(loss.item()):
            return float("nan"), n_params
        loss.backward()
        opt.step()
    return evaluate(model, Xte, yte), n_params


def train_ppr(seed, lr, label_noise=0.0, epochs=EPOCHS):
    Xtr, ytr, Xte, yte = get_data(seed, label_noise)
    model, n_params = make_model(seed)
    crit = nn.CrossEntropyLoss()
    opt = PPR(
        model.parameters(), lr=lr, momentum=0.9,
        tentacles=True, reincarnation=True,
        probe_dirs=8, reach_frac=0.05, pull_force=0.3, pull_steps=20,
        accept_tau=1e-3, progress_tau=5e-3,
        stall_eps=1e-4, stall_steps=10, settle_steps=8,
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

    for _ in range(epochs):
        loss = opt.step(closure, loss_only)
        if not math.isfinite(loss.item()):
            return float("nan"), n_params, opt.stats()
    return evaluate(model, Xte, yte), n_params, opt.stats()


def stat(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return dict(mean=round(mean, 4), std=round(var ** 0.5, 4))


def main():
    t0 = time.time()
    results = {}
    n_params = None

    print("=== normal condition ===")
    for kind, lr in [("sgd_momentum", 0.1), ("adam", 0.01), ("adamw", 0.01)]:
        accs = []
        for s in SEEDS:
            acc, n_params = train_baseline(kind, s, lr)
            accs.append(acc)
        results[f"{kind}_normal"] = stat(accs)
        print(f"{kind}_normal", results[f"{kind}_normal"])

    ppr_accs, ppr_stats_all = [], []
    for s in SEEDS:
        acc, n_params, st = train_ppr(s, 0.1)
        ppr_accs.append(acc)
        ppr_stats_all.append(st)
    results["ppr_normal"] = stat(ppr_accs)
    total_probes = sum(s["probes"] for s in ppr_stats_all)
    total_success = sum(s["successful_pulls"] for s in ppr_stats_all)
    results["ppr_probe_activity"] = f"{total_success} successful / {total_probes} total probes"
    results["n_params"] = n_params
    print("ppr_normal", results["ppr_normal"])
    print("ppr_probe_activity", results["ppr_probe_activity"], " (n_params =", n_params, ")")

    print("\n=== stress condition: high LR + 20% label noise ===")
    for kind, lr in [("sgd_momentum", 2.0), ("adam", 0.5), ("adamw", 0.5)]:
        accs = []
        for s in SEEDS:
            acc, _ = train_baseline(kind, s, lr, label_noise=0.2)
            accs.append(acc)
        results[f"{kind}_stress"] = stat(accs)
        print(f"{kind}_stress", results[f"{kind}_stress"])

    ppr_stress_accs = []
    for s in SEEDS:
        acc, _, _ = train_ppr(s, 2.0, label_noise=0.2)
        ppr_stress_accs.append(acc)
    results["ppr_stress"] = stat(ppr_stress_accs)
    print("ppr_stress", results["ppr_stress"])

    with open("exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
