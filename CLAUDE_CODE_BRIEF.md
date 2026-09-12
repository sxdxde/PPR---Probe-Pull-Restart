# Project brief: PPR optimizer — fix probing, then benchmark

Hand this whole file to Claude Code as the task description. It assumes a machine with a GPU, internet access to download standard datasets, and PyTorch installed (`pip install torch torchvision`).

## Context — read this first

`ppr_optimizer.py` (attached) is a working `torch.optim.Optimizer` implementing Probe–Pull–Restart: momentum descent, augmented with a stall-triggered directional probe-and-pull escape mechanism, and a failure-triggered checkpoint restart ("reincarnation"). Full math and design rationale are in `probe-pull-restart-proposal.pdf` (attached).

**Sandbox testing already found a real problem, and you should not re-litigate it — build on it:**
- On the Rastrigin function, tentacle probing gives a genuine ~2x improvement over plain momentum at 2-D and 5-D, but probe success rate falls to exactly 0% by 20-D (0/496 probes succeeded).
- On a real 2,410-parameter MLP on sklearn `digits`, the stall detector never fired once in 5 seeds × 60 epochs — PPR was indistinguishable from plain momentum.
- A first attempt at a fix (gradient-history-subspace probing instead of isotropic random directions) did **not** rescue the 20-D case.

Full numbers are in `RESULTS.md` (attached). **Do not skip straight to CIFAR-10 training** — at current parameter counts (tens of thousands to millions), the probing mechanism is very likely to be completely inactive, exactly like the digits result, and any benchmark run would just be measuring plain momentum with extra overhead. Fixing this is priority zero.

## Task 1 — make probing actually fire on a real network (do this first, cheaply)

Before any CIFAR-10 run, get probing to succeed at least sometimes on something bigger than 5-D. Try these, cheaply, on the existing `digits` MLP or a slightly bigger one, before committing to a specific fix:

1. **Restrict probing to a subset of parameters**, not the whole model — e.g., only the final linear layer's weights, the way GDSolver (arXiv:2207.03264, cited in the proposal) restricts its MILP search to final-layer weights only. This directly reduces the effective probing dimension without touching the rest of the network.
2. **Increase `probe_dirs` adaptively** with a budget that scales sub-linearly with parameter count (e.g., `probe_dirs = min(64, 4 * sqrt(n_params))`), and re-measure probe success rate at each network size to find where it stops collapsing.
3. **Try true low-rank projection** rather than raw gradient-history combinations: maintain a running SVD (or just power-iteration for the top few directions) of recent gradients, and probe along the actual top eigenvectors rather than random combinations of raw gradient vectors. This is a stronger version of what was already tried and failed — worth testing before concluding the whole gradient-informed direction is a dead end.
4. **Report probe success rate as a function of parameter count** for whichever fix is tried, the same way `dimensionality_collapse.png` did for Rastrigin. If none of these get success rate meaningfully above zero on a network with >10k parameters, that is itself the answer to Research Question 1, and is worth stopping and reporting rather than proceeding to Task 2.

## Task 2 — benchmark, only once Task 1 shows probing is actually active

### Datasets, in order of increasing cost

| Dataset | Why | Cost |
|---|---|---|
| **Synthetic (Rastrigin/Ackley/Rosenbrock, `torch` only)** | Already used; keep as the fast regression test for any change to the probing mechanism. No download. | Seconds |
| **FashionMNIST** (`torchvision.datasets.FashionMNIST`) | Harder than MNIST, still small (28×28 grayscale, 10 classes), trains in minutes even on CPU. Good second checkpoint between `digits` and CIFAR. | Minutes |
| **CIFAR-10** (`torchvision.datasets.CIFAR10`) | The standard benchmark in the optimizer literature (used by Lookahead, SAM, and both SPGD's and Hill-ADAM's closest relatives) — needed for any claim to be taken seriously against those baselines. Use ResNet-18 or a WideResNet. | ~15–40 min/run on one GPU |
| **CIFAR-100** | Same setup, more classes, harder — useful second confirmation once CIFAR-10 results exist. | Similar to CIFAR-10 |
| **CIFAR-10-N** (real human label-noise labels, not synthetic) or synthetic symmetric label noise on CIFAR-10 | For the robustness/stress phase specifically — real noisy-label benchmarks exist and are better-precedented than inventing your own noise injection. | Same as CIFAR-10 |

Do not jump to ImageNet or anything larger — nothing in the proposal's claims requires it, and it multiplies compute cost for no added evidence.

### Baselines (exact optimizers, not just names)

- `torch.optim.SGD(momentum=0.9)`
- `torch.optim.Adam`
- `torch.optim.AdamW`
- Lookahead (wrap any of the above; reference implementation: `https://github.com/michaelrzhang/lookahead`)
- SAM (reference implementation: `https://github.com/davda54/sam`)
- If reproducible within a day of effort: GDSolver, SPGD, or Hill-ADAM (repos linked from their arXiv pages, cited in the proposal's bibliography) — if none reproduce easily, note that in the write-up rather than spending more than a day on it.

### Metrics to log for every run

- Final test accuracy, mean and standard deviation across **≥ 5 seeds** (not 1 — the whole point of this study is that PPR's claimed advantage is about reliability, not peak performance).
- Steps/epochs to reach a fixed target accuracy.
- Wall-clock training time relative to Adam on the same hardware.
- Peak memory relative to Adam (the checkpoint in `_reincarnate` copies the full parameter vector — measure what that actually costs at this scale).
- Probe success rate over training (from `optimizer.stats()`), logged every N steps — this number is the single most important diagnostic and should appear in every plot.

### Stress/robustness runs (Research Question 2)

Repeat a subset of the above with:
- Learning rate deliberately raised until SGD and Adam visibly destabilize (find this threshold empirically per optimizer, the way `MOM_LR`/`ADAM_LR` were tuned empirically in `exp1_rastrigin.py` — do not guess a fixed multiplier).
- 20% symmetric label noise (or CIFAR-10-N labels).
- Synthetic loss spikes: every ~200 steps, replace one minibatch with a random-label batch, modeling the PaLM scenario cited in the proposal.

The claim under test is specifically: **does PPR degrade more gracefully than its baselines under these three conditions**, not whether it's faster under normal conditions.

### Ablations (run alongside the main comparison, not after)

- Tentacles only vs. reincarnation only vs. both vs. neither — same structure as `exp1_rastrigin.py`, at network scale.
- Sustained pull (current design) vs. a single instantaneous kick of matched total impulse — this tests whether the steady-state argument in the proposal (§3.6, Eq. 16) actually matters at this scale or was an artifact of the toy landscape's specific geometry.
- Fixed vs. adaptive reach (`reach_growth`/`reach_cap` on vs. off).

## Deliverables

1. Updated `ppr_optimizer.py` with whichever Task 1 fix worked, plus a short note on which alternatives were tried and rejected, and why.
2. A results table and at least one plot per dataset: accuracy (mean ± std) per optimizer, normal vs. stress conditions, and a probe-success-rate-vs-parameter-count plot analogous to `dimensionality_collapse.png`.
3. An updated `RESULTS.md` that states plainly, for each of the three research questions in the proposal, whether the evidence collected supports, contradicts, or remains inconclusive on it — following the same direct style as the existing `RESULTS.md`, not hedged marketing language.
