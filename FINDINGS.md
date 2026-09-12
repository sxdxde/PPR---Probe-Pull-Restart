# PPR Optimizer — Running Findings

Cumulative, plain-language record of what every experiment actually showed.
Updated after each experiment. Raw numbers live in RESULTS.md and the
individual `exp*_results.json` files. This file is the synthesis.

---

## What PPR is

A PyTorch optimizer that wraps SGD+momentum with two added mechanisms:

- **Tentacle probing**: when the optimizer stalls (velocity RMS < `stall_eps`
  for `stall_steps` consecutive steps), it fires a small number of probe
  directions from the current position, checks whether any of them land in a
  lower-loss region, and if so "pulls" the optimizer toward that direction for
  `pull_steps` steps.

- **Reincarnation**: when the optimizer fails to make progress across
  `max_attempts` consecutive stall events, it restores the best checkpoint
  seen so far (plus a tiny jitter) and resets velocity. Designed for recovery
  from divergence events, not stalls.

---

## Finding 1 — Tentacle probing works at low dimension

**Experiment**: Rastrigin function, dim ∈ {2, 5, 20}, 8 seeds, 3000 steps.  
**Result**: PPR halves the converged loss vs. plain momentum at 2D (2.53 vs
12.81) and 5D (21.39 vs 43.03). At 20D: identical to momentum (163.30 vs
163.30). Probe success rate: 4.8% at 2D, 2.9% at 5D, 0.0% at 20D.  
**Interpretation**: The mechanism genuinely works — a sustained pull toward a
probed better basin is the reason, not noise. But it collapses with dimension.

---

## Finding 2 — Reincarnation alone adds nothing (so far)

**Across all experiments**: every run with reincarnation-only produced results
identical to plain momentum. This is expected: reincarnation is a recovery
mechanism, not a search mechanism. It needs a genuine divergence event — a
loss spike that pushes the optimizer far from its best position — to have
anything to recover from. None of the stall-based experiments (Rastrigin,
digits normal) create that scenario. **Untested under its target condition
(loss spikes) until exp4b.**

---

## Finding 3 — The dimensionality collapse: root cause and fix

**Problem identified**: All early fix attempts (isotropic random, subset
random, sqrt(n)-scaled direction budget, gradient-history PCA) failed for the
same hidden reason. They all probed directions that move **multiple
coordinates simultaneously**. The probability of all k coordinates
simultaneously landing in better basins is (p_1D)^k — exponentially small.
Even restricting to 8 coordinates and using random directions within that
subspace still has this problem.

**Fix — `probe_mode="coord_align"`**: probe ±reach along each coordinate
axis independently. Each probe moves exactly one parameter. Success
probability per probe ≈ p_1D, **independent of total parameter count n**.

**Results** (exp1c, dim sweep to 15,000 parameters):

| Dimension | Baseline (random) | coord_align |
|---|---|---|
| 20 | 0.0% | 63% |
| 100 | 0.0% | 93% |
| 1,000 | 0.0% | 42% |
| 5,000 | 0.0% | 43% |
| 15,000 | 0.0% | **91%** |

The mechanism is now mechanically active at any tested dimension.
The absolute loss improvement at high dimension is small (0.03% at 15k-D)
because only 8 of 15,000 Rastrigin coordinates are probed per stall event —
geometrically expected, not a failure.

---

## Finding 4 — Full-batch networks: probing fires and does no harm

**Experiment**: digits MLP (2,410 params), full-batch, 150 epochs, 5 seeds
(exp3). coord_align with stall_eps=0.01, probe_subset_size=330 (full last
layer).

**Normal condition**:

| Optimizer | Accuracy |
|---|---|
| SGD+momentum | 96.40% ± 1.13% |
| Adam | 96.76% ± 0.51% |
| AdamW | 96.76% ± 0.46% |
| PPR coord_align | 96.36% ± 0.56% |
| PPR probes | 20/20 fired, **100% success** |

PPR matches SGD exactly. Probing is genuinely active and succeeds every time.
It doesn't improve accuracy — digits doesn't create the conditions PPR is
designed to escape (genuine, escapable local minima). Full-batch training is
the correct operating regime for this mechanism.

---

## Finding 5 — Stress condition: no resilience advantage

**Same digits experiment, stress condition**: SGD lr=2.0, Adam lr=0.5,
PPR lr=2.0, + 20% label noise.

| Optimizer | Accuracy |
|---|---|
| SGD+momentum | 13.78% ± 10.18% |
| Adam | 11.24% ± 5.59% |
| AdamW | 12.13% ± 7.56% |
| PPR coord_align | 17.47% ± 11.50% |

All four collapse to near-random (10% baseline for 10 classes). PPR's probe
success rate drops to 5% under high LR: the stall detector barely fires
because high LR keeps velocity large. When stalls do fire, the chaotic
gradient landscape prevents probes from finding better directions.
**No stress resilience from probing under high LR + label noise.**

Caveat: reincarnation theoretically could help here (restore best checkpoint
after the LR drives the optimizer away from it), but the stall-based trigger
doesn't match the failure mode of high-LR chaos.

---

## ⚠️  RETRACTED: Findings 6, 7, and the original Finding 9 were caused by a bug

All MNIST PPR experiments (exp3, exp4a, exp4b, and the first exp5 run) had a
missing `loss.backward()` call in the training closure. PPR's `step()` reads
gradients via `_flat_grad()` after the closure runs — the closure must call
`backward()`. The baselines (SGD, Adam, AdamW) used a separate loop that
explicitly called `backward()`, so they were unaffected. PPR ran with zero
gradients throughout. The model never learned — it sat at random initialization
(10–16% accuracy for 10 classes). Every piece of analysis about "structural
incompatibility with mini-batch training," "velocity-based stall detection
firing on gradient noise," and "best_params being corrupted by mini-batch noise"
was a post-hoc rationalization of what was actually just a missing method call.
The corrected results appear in Finding 9 below.

## Finding 6 — Mini-batch training: PPR catastrophically disrupts learning

**Experiment**: MNIST, 101,770-param MLP, mini-batch=256, 20 epochs, 3 seeds
(exp3). PPR params: stall_eps=0.01, stall_steps=8.

| Optimizer | Accuracy |
|---|---|
| SGD+momentum | 97.89% ± 0.11% |
| Adam | 97.94% ± 0.07% |
| AdamW | 98.02% ± 0.03% |
| PPR (stall_eps=0.01) | **10.76% ± 1.29%** |

**Cause**: 205 stall events per run over 4,680 training steps (one every
~23 steps). With 15 steps in pull+cooldown per event, **65.7% of training
steps were not doing gradient descent**. The network never converged.

Two compounding failure modes:
1. **False stalls**: `stall_steps=8` is too short for mini-batch training.
   Individual mini-batches have different gradient directions; velocity
   temporarily drops below `stall_eps` on "quiet" batches even during
   productive training. This is gradient noise, not a genuine stall.
2. **False probe success**: the probe evaluated on the same mini-batch that
   just ran a training step. With `accept_tau=1e-4`, almost any single-
   coordinate move trivially reduces that batch's loss by ≥0.0001. 615/615
   probes "succeeded" — all false positives. Pull directions don't generalize.

**This is primarily a tuning problem (stall_steps too small), not a
structural one.** See Finding 7.

---

## Finding 7 — Tuned PPR on MNIST still fails (exp4a)

All four targeted fixes applied: stall_steps=200, accept_tau=0.01, pull_steps=5,
pull_force=0.1, probe evaluated on fixed 1,000-sample held-out batch.

| Optimizer | Accuracy |
|---|---|
| SGD+momentum | 97.74% ± 0.09% |
| Adam | 97.92% ± 0.15% |
| AdamW | 97.89% ± 0.11% |
| PPR tuned | **13.60% ± 2.73%** |

PPR tuned: 16 probes (down from 205 — stall_steps=200 worked as intended),
all 16/16 "succeeded", 9 reincarnations across 3 seeds. Still random-guessing.

**Why it still fails — the root cause is structural, not a tuning problem.**

Two compounding issues that parameter tuning cannot fix:

**Issue 1 — stall detection cannot be calibrated for mini-batch training.**
With lr=0.01 and momentum=0.9, equilibrium velocity ≈ lr×|grad|/(1−momentum).
For MNIST gradients (~0.05–0.1 RMS), equilibrium sits at ~0.005–0.01 — right at
or below stall_eps=0.01 — throughout the entire run. Velocity never rises above
the threshold; the stall counter increments continuously. Stall fires like
clockwork every 200 steps (200 stall_steps + 10 pull+settle = 210 steps/cycle,
3510 total steps / 210 = 16.7 events → 16 observed). This is not stall detection;
it is periodic sampling every 210 steps regardless of actual training state.
No value of stall_eps avoids this: lower → never fires; higher → fires faster.
The only structural fix is to detect stalls via a different signal (loss plateau
on a fixed evaluation set, not instantaneous velocity).

**Issue 2 — `best_params` is corrupted by mini-batch noise.**
In `step()`, `best_params` is updated whenever a new minimum mini-batch loss is
seen. In mini-batch training, minimum mini-batch loss ≠ best overall model. An
early mini-batch containing mostly easy samples can show anomalously low loss
even for a barely-trained model. That state becomes `best_params`. Reincarnation
then restores this early, under-trained checkpoint (+ jitter=1e-4, essentially
unchanged). With 3 reincarnations each resetting training to this corrupted
checkpoint, the model cannot converge: it keeps being reset to an early state,
re-training, stalling again at step ~840, re-reincarnating.

**What would actually fix this (a code change, not parameter tuning):**
- Maintain a SEPARATE stable evaluator (large fixed subset, e.g., 5,000 samples)
  used for: (a) tracking best_loss/best_params, (b) accept_tau probe evaluation,
  (c) progress_tau episode evaluation.
- The mini-batch closure is used only for gradient computation.
- This separates "how to train" from "how to judge quality" — the current design
  conflates both, which is fine for full-batch but broken for mini-batch.

**Verdict: PPR is structurally incompatible with mini-batch stochastic training
as currently designed.** The failure is in the optimizer's internal quality
metric (mini-batch loss), not in the probe direction or step sizes. Tuning
parameters cannot fix a measurement problem.

---

## Finding 8 — Reincarnation under loss spikes (exp4b)

First test of reincarnation in its intended scenario. Setup: digits, 250
steps, 4 spikes injected (steps 50/100/150/200) by replacing one training
step's labels with random labels. Conditions: plain momentum, reincarnation-
only (tentacles=False), PPR full (coord_align + reincarnation). 5 seeds.

**Results**:

| Condition | Accuracy | Reinc/run | Recovery steps per spike |
|---|---|---|---|
| Plain momentum | 96.67% ± 1.05% | — | 0, 0, 0, 0 (all seeds) |
| Reincarnation only | 96.31% ± 1.12% | **12.0** | 0, 0, 0, 0 |
| PPR full | 96.76% ± 1.06% | 1.2 | 0, 0, 0, 0 |

**The spikes were not strong enough to test reincarnation.** Recovery = 0
steps for plain momentum on every spike and every seed means plain momentum
already recovered within one training step — before the stall detector (15
steps needed) could even register a problem. A single random-label gradient
step at lr=0.1 produces a perturbation that the very next true-gradient step
corrects. Reincarnation never had a window to act.

**Secondary finding**: reincarnation-only fires 12 times per run during
normal convergence (not spike recovery). With stall_eps=0.01 and
stall_steps=15, the detector triggers as the model naturally converges and
velocity settles — reincarnation then restores best_params, adds tiny
jitter (1e-4), and the cycle repeats. This is a stall calibration issue,
not spike recovery, and it slightly hurts accuracy (96.31% vs 96.67%).
PPR full (with probing) fires reincarnation only 1.2 times because probing
handles most stall events first (exhausts max_attempts before reincarnating).

**What this experiment does not answer**: whether reincarnation helps after
genuinely destabilizing spikes — spikes severe enough that plain momentum
takes >15 steps to recover. To test that, spikes need to be either:
(a) multi-step (5–10 consecutive random-label steps instead of 1), or
(b) paired with a high learning rate that amplifies the perturbation, or
(c) applied to a task where the optimizer is already near a saddle and
    a spike can push it into a worse basin permanently.
**That experiment has not been run.** Reincarnation's value in its target
scenario (genuine long-duration divergence) remains an open question.

---

## What the parameters actually do (tuning guide)

| Parameter | Effect | Direction for real networks |
|---|---|---|
| `stall_eps` | Velocity threshold to declare a stall | Start low; if stalls never fire, raise slightly. For mini-batch: needs calibration against batch gradient scale |
| `stall_steps` | Consecutive low-velocity steps before stall fires | **Critical for mini-batch**: 8 causes noise-triggered stalls; 200 prevents them. Full-batch: 15–30 is fine |
| `accept_tau` | Probe must beat current loss by this much | Too small → false-positive probes on noisy mini-batches; 0.01 is more appropriate than 1e-4 |
| `pull_force` | Strength of pull toward best probe direction | Reduce for mini-batch (0.1 vs 0.3) so gradient still has influence during pull |
| `pull_steps` | How many steps to pull | 5 causes less disruption than 10; reduces blast radius per stall event |
| `probe_subset_size` | How many coordinates to probe (coord_align) | Full last-layer (330 for digits, 1290 for MNIST-MLP) covers the most meaningful parameters at low cost |
| `reach_frac` | Probe step size as fraction of ‖θ‖ | Scale-adaptive; 0.05 (5%) is reasonable; too large overshoots basins |
| `jitter` | Reincarnation noise scale | Must be large enough to escape the exact local minimum; too small → identical restart |

---

## Finding 9 — With the bug fixed, PPR matches SGD on MNIST (exp5, corrected)

**The bug**: every MNIST PPR closure was missing `loss.backward()`. PPR's
`step()` reads gradients via `_flat_grad()` — the closure must call backward.
Baselines (SGD/Adam/AdamW) used a separate training loop with explicit backward,
so they were unaffected. PPR was running with zero gradients; the model sat at
random initialization (~10–16% for 10 classes) the whole time.

**Corrected results** (exp5_plateau.py, backward fixed, 3 seeds, 15 epochs):

| Optimizer | MNIST accuracy | Probes |
|---|---|---|
| SGD+momentum | 97.74% ± 0.09% | — |
| Adam | 97.92% ± 0.15% | — |
| AdamW | 97.89% ± 0.11% | — |
| PPR-plateau | **97.75% ± 0.08%** | 0/1 (1 fired, failed) |
| PPR-velocity (digits) | 96.36% ± 0.56% | 20/20 success |

**PPR-plateau on MNIST matches SGD exactly and does no harm.** The plateau
stall detector correctly identifies that MNIST training is making genuine
progress and almost never fires (only 1 probe across all 3 seeds, which
found nothing better and correctly rejected). Zero reincarnations. Zero
disruption.

**Why PPR doesn't fire on MNIST**: With stall_eps=5e-4 and stall_steps=100,
a stall requires the eval EMA to not improve by 0.0005 for 100 consecutive
steps (~0.4 epochs). MNIST with SGD at lr=0.01 converges smoothly — the
loss drops continuously enough that the plateau detector never triggers in
15 epochs. This is correct behavior: PPR should not fire when the base
optimizer is making real progress.

**What this means for "does PPR have potential"**:
- On easy, smoothly-converging tasks (MNIST, digits): PPR is neutral — it
  correctly stays dormant and matches the base optimizer.
- To see the mechanism activate usefully, we need a task where the optimizer
  genuinely stalls in a suboptimal basin — a harder landscape than MNIST.
  The Rastrigin results (2D–5D) already showed this is possible in principle.
- The correct next test is a task that is actually hard enough for the base
  optimizer to stall: CIFAR-10 with a ResNet is the right candidate, but
  requires a GPU. On CPU-feasible tasks tested here, the mechanism is correct
  but has nothing to do.

---

## Finding 10 — Strong loss spikes on MNIST (exp6)

10 consecutive random-label gradient steps at lr=0.05 (5× normal), injected
4 times during training at steps 1000/2000/3000/4000. MNIST, 101K-param MLP,
3 seeds, 20 epochs. Conditions: no_spike (ceiling), plain_momentum+spikes,
reinc_only+spikes, ppr_full+spikes.

| Condition | Accuracy | Reinc/run |
|---|---|---|
| No spikes (ceiling) | 97.89% ± 0.11% | 0 |
| Plain momentum + spikes | 97.88% ± 0.09% | 0 |
| Reincarnation only + spikes | 97.66% ± 0.24% | 23.3 |
| PPR full + spikes | **97.89% ± 0.11%** | 2 |

**Plain momentum recovers completely from every spike.** 97.88% vs 97.89%
clean — within noise. The spikes are real (post-spike eval loss jumps 2–3×),
but MNIST's gradient signal is strong enough that momentum recovers within
~100–200 training steps (less than one epoch) without any help. Reincarnation
never had a window to act, not because the spike was too weak, but because
plain momentum's own momentum state (still pointing toward the true minimum
at the time of the spike) carries it through before the plateau detector
could even fire.

**Reincarnation-only is slightly worse than plain momentum** (97.66% vs 97.88%).
With stall_steps=50, the plateau detector fires 20–26 times per run — 4 of
those are post-spike stalls, the rest are from normal late-training convergence.
Each non-spike reincarnation is unnecessary noise. The overhead of restoring
best_params + velocity reset 20× per run slightly degrades final accuracy.

**PPR full matches clean training exactly** (97.89%). The probing layer handles
most stall events (19–20 probes per run, 1–3 probe successes), leaving only 2
reincarnations per run. Probing acts as a buffer: it absorbs stall events
gracefully before reincarnation escalates. This is the cleanest result for PPR
full — neutral on a smooth task, not harmful.

**Why reincarnation can't help here even with strong spikes:**
MNIST is a robust, easy task. At any point in training, the gradient of the
true loss is a strong, low-noise signal that points toward the minimum. When
a spike pushes the model off, the gradient immediately starts pulling it back.
Reincarnation would only add value if recovery took longer than stall_steps
consecutive steps — i.e., if the post-spike gradient signal were weak or noisy.
That happens in: (a) near-converged training on a hard task where gradients
are tiny, or (b) a task with sharp local minima where the spike lands the model
in a bad basin it can't climb out of. MNIST doesn't have either property.

**What would actually test reincarnation's value:**
- Spikes applied at epoch 18–20 of 20 (when MNIST gradients are tiny, < 1e-4),
  so recovery takes 500+ steps rather than 100
- A harder task (CIFAR-10/ResNet) where the loss landscape has real structure
  and post-spike recovery is not guaranteed by gradient descent alone
- Synthetic tasks with explicit local minima (Rastrigin at scale) — already
  showed coord_align works there; reincarnation untested in that context

---

## Open questions

1. **Does reincarnation help on a hard task near convergence?** The mechanism
   needs a scenario where post-spike gradient signal is too weak for plain
   momentum to self-recover. CIFAR-10 at epoch 90/100 is the right test.
   Requires GPU.

2. **Does coord_align probing improve accuracy on a task where SGD genuinely
   stalls?** On MNIST/digits, SGD never stalls — PPR stays dormant. On CIFAR-10
   with a ResNet, SGD does plateau in late training. Requires GPU.

3. **PPR + SAM**: SAM finds flat minima within basins; PPR escapes between
   basins. Complementary in theory. Not testable until probing activates on
   a real hard task first. Requires GPU.

4. **Reincarnation on Rastrigin at scale**: coord_align probing is validated
   at high dimension. Reincarnation has never been tested on the same landscape.
   Would require injecting a spike into the Rastrigin optimizer run — CPU-feasible.
