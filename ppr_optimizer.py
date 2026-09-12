"""
Probe-Pull-Restart (PPR) optimizer.

A PyTorch implementation of the mechanism specified in
`probe-pull-restart-proposal.pdf`, generalized from the 1-D reference
implementation to R^n following the "Phase 2" design in that document:
fixed coordinate offsets (which only make sense in 1-D) are replaced by
a small number of random unit directions in parameter space, redrawn at
every stall event.

Because the probing step needs extra loss *evaluations* (not gradients),
this optimizer follows PyTorch's closure convention (the same pattern
used by torch.optim.LBFGS), but splits it into two closures:

    - `closure()`      : zero_grad + forward + backward, returns the loss
                          tensor. Used once per step for the main update.
    - `loss_only_fn()`  : forward pass only, no backward, returns a float.
                          Used for each probe evaluation, which is why
                          probing is gated behind the stall detector --
                          it must stay rare, since each probe direction
                          costs one extra forward pass.

probe_mode choices
------------------
  "random"           : k random unit directions in full (or subset) space.
                       Fails at high dimension: success probability falls
                       as (p_1D)^k because all k coords move simultaneously.

  "grad_subspace"    : random linear combos of recent gradient vectors.
  "grad_pca"         : true top singular vectors of recent gradient history.
                       Both also collapse at high dim (same joint-improvement
                       problem; confirmed in exp1b -- see RESULTS.md).

  "coord_align"      : axis-aligned probing on a subset of coordinates.
                       Each probe moves exactly ONE coordinate by ±reach.
                       Success probability per probe is ~p_1D INDEPENDENT
                       of total parameter count -- decouples the dimensions.
                       Requires probe_subset_size or probe_indices to be set.
                       Total probes per stall: 2 * len(probe_indices).

  "coord_align_multi_reach" : same axis-aligned coordinate probing, but
                       sweeps four reach scales (0.5×, 1×, 1.5×, 2×) per
                       direction to reduce sensitivity to basin spacing.
                       Total probes per stall: 8 * len(probe_indices).

Example
-------
    optimizer = PPR(model.parameters(), lr=0.01, momentum=0.9)

    def closure():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        return loss

    def loss_only():
        with torch.no_grad():
            return criterion(model(x), y).item()

    loss = optimizer.step(closure, loss_only)

Task 1 dimensionality-fix note (see RESULTS.md and exp1b_dimensionality_fix.py)
---------------------------------------------------------------------------
Three fixes for the probe-success collapse at high dimension were implemented
here and tested (`probe_indices` / `probe_subset_size`, and `probe_mode=
"grad_pca"`; sqrt(n)-scaled `probe_dirs` needed no code change). All three
were REJECTED: none raised probe success rate meaningfully above 0% at
15,000 parameters (>10k, the brief's bar), including subset-restricted
probing, which still fixes the probed subspace at a constant 8 dimensions
regardless of total parameter count. Verified this isn't a float32 precision
artifact by rerunning in float64 (result: 1/310 successes vs. 0/310, still
~0%). The parameters below remain available (default off / no-op) as
documented, rejected options -- not because they're recommended, but because
ripping them out would lose the evidence that they were tried.

The root cause confirmed by analysis: all three failed fixes still used
directions that move multiple coordinates simultaneously (random combos in the
subset subspace). The probability of ALL k subset coordinates simultaneously
landing in better basins is (p_1D)^k -- exponentially small. "coord_align"
probing (see above) addresses this by testing one coordinate at a time.
"""

import torch
from torch.optim import Optimizer


class PPR(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        # --- tentacle probing (search) ---
        tentacles: bool = True,
        probe_dirs: int = 6,
        probe_mode: str = "random",     # "random", "grad_subspace", or "grad_pca"
        grad_history: int = 10,         # rank of the gradient-history subspace, if used
        probe_indices=None,             # 1-D LongTensor: restrict probing to these flat-param
                                         # indices (e.g. the last layer's weights, GDSolver-style).
                                         # None = probe the full parameter vector.
        probe_subset_size: int = None,  # if set and probe_indices is None, auto-restrict probing
                                         # to the LAST probe_subset_size flat-param coordinates.
        reach_frac: float = 0.02,       # probe distance, as a fraction of ||x||
        reach_abs: float = None,        # if set, overrides reach_frac with a fixed absolute reach
        pull_force: float = 0.6,
        pull_steps: int = 45,
        accept_tau: float = 1e-4,       # per-probe acceptance threshold
        progress_tau: float = 1e-3,     # episode-progress threshold (must be > accept_tau)
        reach_growth: float = 1.18,
        reach_cap: float = 3.2,
        # --- stall detection (predictive node) ---
        stall_eps: float = 1e-3,
        stall_steps: int = 40,
        settle_steps: int = 20,
        # --- stall mode ---
        # "auto"    : use plateau detection when eval_fn is passed to step(),
        #             fall back to velocity-based when it is not.
        # "velocity": always use velocity-based (original behaviour).
        # "plateau" : always use plateau detection (eval_fn required in step()).
        #
        # Plateau detection reinterprets stall_eps as the minimum EMA-loss
        # improvement that counts as progress.  When the EMA has not improved
        # by stall_eps for stall_steps consecutive steps, a stall is declared.
        # This is robust to mini-batch gradient noise because it tracks a
        # smoothed quality signal, not instantaneous velocity.
        stall_mode: str = "auto",
        loss_ema_alpha: float = 0.98,   # EMA smoothing (higher = slower EMA)
        # --- reincarnation (past node / recovery) ---
        reincarnation: bool = True,
        max_attempts: int = 6,
        jitter: float = 1e-4,
    ):
        if progress_tau <= accept_tau:
            raise ValueError(
                "progress_tau must be > accept_tau, or the reach/attempt "
                "counters never advance (see proposal Sec. 3.6 for why)."
            )
        defaults = dict(lr=lr, momentum=momentum)
        params = list(params)
        super().__init__(params, defaults)

        device = params[0].device
        n = sum(p.numel() for p in params)

        if probe_indices is None and probe_subset_size is not None:
            k = min(probe_subset_size, n)
            probe_indices = torch.arange(n - k, n, device=device)
        if probe_indices is not None:
            probe_indices = torch.as_tensor(probe_indices, dtype=torch.long, device=device)

        self.hp = dict(
            tentacles=tentacles, probe_dirs=probe_dirs, probe_mode=probe_mode,
            grad_history=grad_history, probe_indices=probe_indices,
            reach_frac=reach_frac, reach_abs=reach_abs,
            pull_force=pull_force, pull_steps=pull_steps,
            accept_tau=accept_tau, progress_tau=progress_tau,
            reach_growth=reach_growth, reach_cap=reach_cap,
            stall_eps=stall_eps, stall_steps=stall_steps, settle_steps=settle_steps,
            stall_mode=stall_mode, loss_ema_alpha=loss_ema_alpha,
            reincarnation=reincarnation, max_attempts=max_attempts, jitter=jitter,
        )
        self.g = dict(
            velocity=torch.zeros(n, device=device),
            status="descending",
            stall_counter=0, attempts=0, reach_mult=1.0,
            best_loss=float("inf"), best_params=None,
            episode_best_loss=float("inf"),
            pull_dir=None, pull_timer=0, settle_timer=0,
            n_reincarnations=0, n_probes=0, n_pulls_succeeded=0,
            grad_buffer=[],
            # plateau stall detection state
            loss_ema=None, stall_ref_loss=None,
        )

    # -------------------- flat parameter helpers --------------------
    def _flat_params(self):
        return torch.cat(
            [p.detach().reshape(-1) for grp in self.param_groups for p in grp["params"]]
        )

    def _set_flat_params(self, vec):
        i = 0
        for grp in self.param_groups:
            for p in grp["params"]:
                num = p.numel()
                p.detach().copy_(vec[i:i + num].view_as(p))
                i += num

    def _flat_grad(self):
        parts = []
        for grp in self.param_groups:
            for p in grp["params"]:
                parts.append(
                    p.grad.detach().reshape(-1) if p.grad is not None
                    else torch.zeros_like(p).reshape(-1)
                )
        return torch.cat(parts)

    # -------------------- main step --------------------
    @torch.no_grad()
    def step(self, closure, loss_only_fn=None, eval_fn=None):
        """
        closure     : zero_grad + forward + backward, returns loss tensor.
        loss_only_fn: forward only, returns float.  Used for probe evaluation.
        eval_fn     : forward only on a FIXED held-out subset, returns float.
                      When provided, activates plateau-based stall detection
                      and uses the stable quality signal for best_params
                      tracking.  If None, the original velocity-based stall
                      detection is used (backward-compatible).
        """
        hp, st = self.hp, self.g
        lr = self.param_groups[0]["lr"]
        momentum = self.param_groups[0]["momentum"]

        # ── gradient step ──────────────────────────────────────────
        with torch.enable_grad():
            loss = closure()
        x = self._flat_params()
        grad = self._flat_grad()

        if hp["probe_mode"] == "grad_subspace":
            st["grad_buffer"].append(grad.clone())
            if len(st["grad_buffer"]) > hp["grad_history"]:
                st["grad_buffer"].pop(0)

        pulling = st["pull_timer"] > 0
        force = -lr * grad
        if pulling:
            force = force + hp["pull_force"] * st["pull_dir"]
        st["velocity"] = momentum * st["velocity"] + force
        self._set_flat_params(x + st["velocity"])

        # ── quality tracking ────────────────────────────────────────
        # Decide which signal to use for best_params and stall detection.
        use_plateau = (eval_fn is not None and hp["stall_mode"] in ("auto", "plateau"))
        quality = eval_fn() if use_plateau else loss.item()

        if quality < st["best_loss"]:
            st["best_loss"] = quality
            st["best_params"] = self._flat_params().clone()

        # ── pull phase ─────────────────────────────────────────────
        if pulling:
            st["pull_timer"] -= 1
            st["status"] = "pulling"
            if st["pull_timer"] == 0:
                self._evaluate_attempt()
            return loss

        # ── settle / cooldown phase ────────────────────────────────
        if st["settle_timer"] > 0:
            st["settle_timer"] -= 1
            st["status"] = "cooldown"
            return loss

        # ── stall detection ────────────────────────────────────────
        if use_plateau:
            # Loss-plateau detection: EMA of quality loss compared to a
            # rolling reference.  Robust to mini-batch gradient noise.
            alpha = hp["loss_ema_alpha"]
            if st["loss_ema"] is None:
                st["loss_ema"] = quality
                st["stall_ref_loss"] = quality
            else:
                st["loss_ema"] = alpha * st["loss_ema"] + (1.0 - alpha) * quality

            improvement = st["stall_ref_loss"] - st["loss_ema"]
            if improvement > hp["stall_eps"]:
                # Genuine progress — reset counter and advance reference.
                st["stall_counter"] = 0
                st["stall_ref_loss"] = st["loss_ema"]
                st["status"] = "descending"
            else:
                st["stall_counter"] += 1
                st["status"] = "stalling"
        else:
            # Original velocity-based detection (full-batch / backward-compat).
            vnorm = st["velocity"].norm().item() / (st["velocity"].numel() ** 0.5)
            st["stall_counter"] = st["stall_counter"] + 1 if vnorm < hp["stall_eps"] else 0
            st["status"] = "descending" if vnorm >= hp["stall_eps"] else st["status"]

        # ── fire stall ─────────────────────────────────────────────
        if st["stall_counter"] >= hp["stall_steps"]:
            st["stall_counter"] = 0
            if use_plateau:
                # Reset the reference so we measure from the current EMA
                # after a stall fires, not the old baseline.
                st["stall_ref_loss"] = st["loss_ema"]
            if hp["tentacles"] and loss_only_fn is not None:
                self._probe(loss_only_fn)
            elif hp["reincarnation"]:
                self._reincarnate()

        return loss

    # -------------------- probing / pulling --------------------
    def _probe(self, loss_only_fn):
        hp, st = self.hp, self.g
        x = self._flat_params()
        n = x.numel()
        base_reach = hp["reach_abs"] if hp["reach_abs"] is not None else hp["reach_frac"] * (x.norm().item() + 1e-8)
        reach = base_reach * st["reach_mult"]

        cur_loss = loss_only_fn()
        idx = hp["probe_indices"]
        best_delta, best_loss = None, None

        if hp["probe_mode"] in ("coord_align", "coord_align_multi_reach"):
            # Axis-aligned coordinate probing: each probe moves exactly ONE coordinate.
            # Success probability per probe ≈ p_1D, independent of total n -- this is
            # the structural fix for the dimensionality collapse (see module docstring).
            coords = idx if idx is not None else torch.arange(
                max(0, n - hp["probe_dirs"]), n, device=x.device
            )
            reach_scales = [0.5, 1.0, 1.5, 2.0] if hp["probe_mode"] == "coord_align_multi_reach" else [1.0]
            for ci in coords:
                c = ci.item() if hasattr(ci, "item") else int(ci)
                for sign in (1.0, -1.0):
                    for scale in reach_scales:
                        delta = torch.zeros(n, device=x.device)
                        delta[c] = sign * reach * scale
                        self._set_flat_params(x + delta)
                        l = loss_only_fn()
                        if best_loss is None or l < best_loss:
                            best_loss, best_delta = l, delta.clone()
        else:
            # Original multi-coordinate direction probing (random / grad_subspace / grad_pca).
            basis = None
            pca_dirs = None
            if hp["probe_mode"] == "grad_subspace" and len(st["grad_buffer"]) >= 2:
                basis = torch.stack(st["grad_buffer"], dim=0)  # (k, n)
            elif hp["probe_mode"] == "grad_pca" and len(st["grad_buffer"]) >= 2:
                g = torch.stack(st["grad_buffer"], dim=0)  # (k, n)
                g = g - g.mean(dim=0, keepdim=True)
                try:
                    _, _, Vh = torch.linalg.svd(g, full_matrices=False)
                    pca_dirs = Vh
                except RuntimeError:
                    pca_dirs = None

            for i in range(hp["probe_dirs"]):
                if pca_dirs is not None:
                    k = pca_dirs.shape[0]
                    sign = 1.0 if (i // k) % 2 == 0 else -1.0
                    d = sign * pca_dirs[i % k]
                elif basis is not None:
                    coeffs = torch.randn(basis.shape[0], device=x.device)
                    d = coeffs @ basis
                else:
                    d = torch.randn(n, device=x.device)
                if idx is not None:
                    mask = torch.zeros(n, device=x.device)
                    mask[idx] = 1.0
                    d = d * mask
                d = d / (d.norm() + 1e-8)
                self._set_flat_params(x + reach * d)
                l = loss_only_fn()
                if best_loss is None or l < best_loss:
                    best_loss, best_delta = l, reach * d

        self._set_flat_params(x)  # restore before deciding
        st["n_probes"] += 1
        st["status"] = "reaching"

        if best_loss is not None and best_loss < cur_loss - hp["accept_tau"]:
            st["pull_dir"] = best_delta / (best_delta.norm() + 1e-8)
            st["pull_timer"] = hp["pull_steps"]
            st["n_pulls_succeeded"] += 1
        else:
            self._evaluate_attempt()

    def _evaluate_attempt(self):
        hp, st = self.hp, self.g
        improved = st["best_loss"] < st["episode_best_loss"] - hp["progress_tau"]
        if improved:
            st["attempts"] = 0
            st["reach_mult"] = 1.0
            st["episode_best_loss"] = st["best_loss"]
        else:
            st["attempts"] += 1
            st["reach_mult"] = min(hp["reach_cap"], st["reach_mult"] * hp["reach_growth"])

        if hp["reincarnation"] and st["attempts"] >= hp["max_attempts"]:
            self._reincarnate()
            st["attempts"] = 0
        st["settle_timer"] = hp["settle_steps"]

    def _reincarnate(self):
        st = self.g
        if st["best_params"] is not None:
            jitter = torch.randn_like(st["best_params"]) * self.hp["jitter"]
            self._set_flat_params(st["best_params"] + jitter)
        st["velocity"] = torch.randn_like(st["velocity"]) * self.hp["jitter"]
        st["n_reincarnations"] += 1
        st["settle_timer"] = self.hp["settle_steps"]
        st["status"] = "reviving"

    # -------------------- introspection --------------------
    def stats(self):
        st = self.g
        return dict(
            status=st["status"], best_loss=st["best_loss"],
            reincarnations=st["n_reincarnations"], probes=st["n_probes"],
            successful_pulls=st["n_pulls_succeeded"], reach_mult=st["reach_mult"],
        )
