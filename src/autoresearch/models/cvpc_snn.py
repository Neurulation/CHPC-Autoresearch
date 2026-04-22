"""Complex-Valued Predictive Coding SNN (CVPC-SNN).

Integrates three ideas into a single spiking model:

  1. **Complex-valued synaptic weights** W = r·exp(jθ), where magnitude r encodes
     synaptic efficacy and phase θ encodes preferred timing delay.

  2. **Temporal (phase) coding**: information is carried in spike *latency* relative
     to a global reference oscillation.  Earlier spike → stronger signal:
       x_i = exp(-α · t_i^f)
     The complex representation r^(l) ∈ ℂ^N combines magnitude (spike strength)
     and phase (timing relative to oscillation).

  3. **Predictive Coding (PC) inference** on the complex representations, with a
     three-factor Complex-STDP weight-update rule:
       dr_ij/dt = η_r · Re[ε_i · Δ_ij]   (magnitude update)
       dθ_ij/dt = η_θ · Im[ε_i · Δ_ij] / r_ij   (phase / delay update)
     where ε_i is the complex prediction error at neuron i and Δ_ij is the STDP
     timing kernel evaluated at the spike-time difference.

Architecture (3-layer hierarchy for MNIST)
------------------------------------------
Input    : (N, 1, 28, 28) → flatten → (N, 784) real
Encode   : real → complex via temporal encoding (spike latency → phase)
Layer 1  : ComplexLIFLayer(784 → 256)
Layer 2  : ComplexLIFLayer(256 → 128)
Output   : ComplexLIFLayer(128 → 10)
PC inf   : T_pc gradient steps on {r^(1), r^(2)} with r^(3) clamped to one_hot(y)
CV-STDP  : weight update via three-factor rule applied once per training step
Classify : linear(128 → 10) on |r^(2)| (spike magnitudes after PC inference)

Training loss
-------------
  F = Σ_l ‖ε^(l)‖²  +  λ_ce · CE(cls_head(|r^(2)|), y)

The CV-STDP rule is applied *after* the gradient-based PC weight update, acting as
a second-order biological correction that adjusts phases.  In the first version, the
PC weight-update energy is backpropagated through the complex linear layers via
PyTorch autograd (Wirtinger derivatives are handled automatically for torch.cfloat).
The STDP rule is then applied as an explicit parameter update on top.

Modes
-----
  'pc_only'   : PC energy + CE on cls_head.  No STDP.  Baseline ablation.
  'stdp_only' : STDP weight update only, no PC inference energy.  Ablation.
  'full'      : PC energy + STDP (default).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: convert real input to complex temporal representation
# ---------------------------------------------------------------------------

def real_to_complex_temporal(
    x: torch.Tensor,
    alpha: float = 3.0,
    omega: float = 1.0,
) -> torch.Tensor:
    """Encode real-valued activations as complex spike representations.

    Maps x ∈ [0,1] to complex r ∈ ℂ by treating x as a normalised spike
    *magnitude* and deriving phase from the inverse latency:
      phase = ω · (1 - x)·π       (x=1 → phase=0, spikes first; x=0 → phase=π)
      magnitude = exp(-α · (1-x))  (x=1 → mag≈1; x=0 → mag≈0)
    Returns r = magnitude * exp(j·phase).

    Args:
        x: Real tensor ∈ [0, 1], shape (...,).
        alpha: Latency decay constant.
        omega: Phase scaling factor.

    Returns:
        Complex tensor of same shape as x.
    """
    x = x.clamp(0.0, 1.0)
    latency = 1.0 - x                         # small x → large latency
    magnitude = torch.exp(-alpha * latency)    # early spikes are stronger
    phase = omega * latency * torch.pi         # phase proportional to latency
    return torch.polar(magnitude, phase)       # magnitude * exp(j*phase)


# ---------------------------------------------------------------------------
# Complex linear layer (operates on torch.cfloat tensors)
# ---------------------------------------------------------------------------

class ComplexLinear(nn.Module):
    """Fully-connected layer with complex-valued weights W = r·exp(jθ).

    Uses a single nn.Linear with dtype=torch.cfloat.  PyTorch's autograd
    handles Wirtinger derivatives automatically for complex parameters.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to include a complex bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialise with small random complex weights (magnitude ~ 0.1, phase uniform)
        mag = torch.empty(out_features, in_features).uniform_(0.0, 0.2)
        phase = torch.empty(out_features, in_features).uniform_(-torch.pi, torch.pi)
        w_init = torch.polar(mag, phase).to(torch.cfloat)
        self.weight = nn.Parameter(w_init)

        if bias:
            b_mag = torch.zeros(out_features)
            b_phase = torch.zeros(out_features)
            self.bias = nn.Parameter(torch.polar(b_mag, b_phase).to(torch.cfloat))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @property
    def weight_magnitude(self) -> torch.Tensor:
        return self.weight.abs()

    @property
    def weight_phase(self) -> torch.Tensor:
        return self.weight.angle()


# ---------------------------------------------------------------------------
# Complex LIF layer (magnitude thresholding on complex potential)
# ---------------------------------------------------------------------------

class ComplexLIFLayer(nn.Module):
    """Complex-valued Leaky Integrate-and-Fire layer.

    Maintains a complex membrane potential U ∈ ℂ^N.  A spike is emitted when
    |U| ≥ threshold.  Post-spike reset: magnitude resets to u_reset, phase preserved.

    For gradient computation we use a *soft* spike function on |U|:
      surrogate(|U|) = sigmoid(slope · (|U| - threshold))
    The complex output r is then surrogate * U/|U| (unit-phase vector scaled by
    the surrogate activation).

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        beta:         Membrane potential leak (0 < beta < 1).
        threshold:    Spike threshold on |U|.
        u_reset:      Post-spike reset magnitude.
        slope:        Surrogate gradient slope.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        u_reset: float = 0.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.u_reset = u_reset
        self.slope = slope
        self.fc = ComplexLinear(in_features, out_features)

    def init_mem(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.fc.out_features, dtype=torch.cfloat, device=device)

    def _surrogate(self, u_mag: torch.Tensor) -> torch.Tensor:
        """Differentiable soft spike function on membrane magnitude."""
        return torch.sigmoid(self.slope * (u_mag - self.threshold))

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep update.

        Args:
            x:   Complex input, shape (N, in_features), dtype cfloat.
            mem: Complex membrane potential, shape (N, out_features), dtype cfloat.

        Returns:
            spk: Complex spike output (surrogate × unit phase), shape (N, out_features).
            mem: Updated membrane potential.
        """
        # LIF dynamics: τ dU/dt = -U + I  →  U[t+1] = β·U[t] + I[t+1]
        cur = self.fc(x)
        mem = self.beta * mem + cur

        u_mag = mem.abs()                          # |U|, shape (N, out)
        u_phase = mem.angle()                       # ∠U, shape (N, out)

        # Surrogate spiking: differentiable approximation
        spk_mag = self._surrogate(u_mag)           # in [0, 1]

        # Represent spike as complex: magnitude = surrogate, phase = membrane phase
        spk = torch.polar(spk_mag, u_phase)

        # Post-spike reset: subtract threshold worth of magnitude, preserve phase
        reset_mag = (spk_mag > 0.5).float() * (u_mag - self.u_reset)
        reset_vec = torch.polar(reset_mag.clamp(min=0.0), u_phase)
        mem = mem - reset_vec

        return spk, mem


# ---------------------------------------------------------------------------
# PC hierarchy helpers
# ---------------------------------------------------------------------------

def _complex_energy(
    r_lower: torch.Tensor,
    r_upper: torch.Tensor,
    layer: ComplexLinear,
    detach_weights: bool,
) -> torch.Tensor:
    """Squared complex prediction error ‖r_lower - f(W·r_upper)‖².

    The prediction is mu = W·r_upper (no nonlinearity at output; tanh-phase
    could be added but is omitted for transparency).

    Args:
        r_lower:        Lower-layer representation (N, d_l), cfloat.
        r_upper:        Upper-layer representation (N, d_{l+1}), cfloat.
        layer:          ComplexLinear that maps r_upper → r_lower.
        detach_weights: If True, detach weight from graph (representation update).
    """
    if detach_weights:
        w = layer.weight.detach()
        b = layer.bias.detach() if layer.bias is not None else None
        mu = F.linear(r_upper, w, b)
    else:
        mu = layer(r_upper)

    eps = r_lower - mu                     # complex error, shape (N, d_l)
    return (eps.abs() ** 2).sum()          # sum of squared moduli


# ---------------------------------------------------------------------------
# Complex STDP (applied as an explicit post-step update)
# ---------------------------------------------------------------------------

class ComplexSTDPUpdater:
    """Three-factor Complex STDP weight updater.

    Applied as a manual parameter update *after* the autograd PC update.
    Works on a single ComplexLinear layer.

    Three-factor rule:
      dr_ij/dt  = η_r · Re[ε_i · Δ_ij]
      dθ_ij/dt  = η_θ · Im[ε_i · Δ_ij] / max(r_ij, eps)

    where:
      ε_i  = complex prediction error at post-synaptic neuron i
      Δ_ij = STDP timing function evaluated at (t_post_i - t_pre_j)

    Phase regulariser (Kuramoto-like, prevents drift):
      dθ_ij/dt -= λ · sin(θ_ij - θ̄_l)   where θ̄_l = mean phase in layer l

    Spike times are approximated from the complex representations:
      t_i ≈ 1 - |r_i| / max(|r|)          (larger magnitude → earlier spike)

    Args:
        eta_r:   Learning rate for magnitude update.
        eta_th:  Learning rate for phase update.
        tau_pos: STDP LTP time constant (steps).
        tau_neg: STDP LTD time constant (steps).
        A_pos:   STDP LTP amplitude.
        A_neg:   STDP LTD amplitude.
        lam:     Phase coherence regulariser strength.
        eps:     Minimum magnitude to avoid division by zero.
    """

    def __init__(
        self,
        eta_r: float = 1e-4,
        eta_th: float = 1e-4,
        tau_pos: float = 20.0,
        tau_neg: float = 20.0,
        A_pos: float = 0.01,
        A_neg: float = 0.01,
        lam: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        self.eta_r = eta_r
        self.eta_th = eta_th
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.lam = lam
        self.eps = eps

    def _spike_times(self, r: torch.Tensor) -> torch.Tensor:
        """Approximate spike times from complex representations.

        t_i ≈ 1 - |r_i| / (max_j |r_j| + eps)   ∈ [0, 1]
        Shape of r: (N, D).  Returns (N, D).
        """
        mag = r.abs()
        max_mag = mag.amax(dim=1, keepdim=True).clamp(min=self.eps)
        return 1.0 - mag / max_mag

    def _stdp_kernel(
        self, t_post: torch.Tensor, t_pre: torch.Tensor
    ) -> torch.Tensor:
        """STDP kernel Δ(t_post - t_pre).

        Args:
            t_post: (N, out) post-synaptic spike times.
            t_pre:  (N, in)  pre-synaptic spike times.

        Returns:
            Δ of shape (N, out, in): signed timing kernel, real-valued.
        """
        # dt[n, i, j] = t_post[n,i] - t_pre[n,j]
        dt = t_post.unsqueeze(2) - t_pre.unsqueeze(1)  # (N, out, in)
        ltp = self.A_pos * torch.exp(-dt.abs() / self.tau_pos) * (dt > 0).float()
        ltd = -self.A_neg * torch.exp(-dt.abs() / self.tau_neg) * (dt <= 0).float()
        return ltp + ltd  # real-valued Δ, shape (N, out, in)

    @torch.no_grad()
    def update(
        self,
        layer: ComplexLinear,
        r_post: torch.Tensor,
        r_pre: torch.Tensor,
        eps_post: torch.Tensor,
    ) -> None:
        """Apply complex STDP update to layer.weight in-place.

        Args:
            layer:    ComplexLinear layer to update.
            r_post:   Post-synaptic representations, shape (N, out), cfloat.
            r_pre:    Pre-synaptic representations,  shape (N, in),  cfloat.
            eps_post: Complex prediction error at post-syn neurons, shape (N, out).
        """
        t_post = self._spike_times(r_post)          # (N, out)
        t_pre = self._spike_times(r_pre)             # (N, in)
        Delta = self._stdp_kernel(t_post, t_pre)     # (N, out, in), real

        # Broadcast eps_post: (N, out) → (N, out, 1) × Delta (N, out, in) → (N, out, in)
        # Product eps_i * Δ_ij:
        eps_Delta = eps_post.unsqueeze(2) * Delta    # complex * real → complex (N, out, in)

        # Batch-mean over N
        eps_Delta_mean = eps_Delta.mean(dim=0)       # (out, in), cfloat

        # Current weight polar form
        w = layer.weight                              # (out, in), cfloat
        r_ij = w.abs().clamp(min=self.eps)
        theta_ij = w.angle()

        # Magnitude update: dr = η_r · Re[ε·Δ]
        dr = self.eta_r * eps_Delta_mean.real        # (out, in)
        r_ij_new = (r_ij + dr).clamp(min=0.0)

        # Phase update: dθ = η_θ · Im[ε·Δ] / r_ij
        dtheta = self.eta_th * eps_Delta_mean.imag / r_ij  # (out, in)

        # Phase coherence regulariser: dθ -= λ · sin(θ_ij - θ̄)
        theta_mean = theta_ij.mean()
        dtheta = dtheta - self.lam * torch.sin(theta_ij - theta_mean)

        theta_ij_new = theta_ij + dtheta

        # Reconstruct complex weight
        layer.weight.copy_(torch.polar(r_ij_new, theta_ij_new).to(torch.cfloat))


# ---------------------------------------------------------------------------
# Main model: CVPCSNNModel
# ---------------------------------------------------------------------------

class CVPCSNNModel(nn.Module):
    """Complex-Valued Predictive Coding SNN for MNIST classification.

    Combines CV-LIF spiking dynamics with a PC inference loop and an optional
    Complex-STDP three-factor weight update.

    Architecture (3-layer hierarchy):
      Input (784) → ComplexLIFLayer(784→256) → r^(1)
                  → ComplexLIFLayer(256→128) → r^(2)
                  → ComplexLIFLayer(128→10)  → r^(3)
      PC inference: T_pc steps on {r^(1), r^(2)} with r^(3) clamped = one_hot(y).
      CV-STDP: applied to each ComplexLinear weight after PC gradient step.
      Classify: cls_head(|r^(2)|) (real-valued linear on spike magnitudes).

    Args:
        input_size:    Flattened input dimension (784 for MNIST).
        hidden_dims:   Hidden layer widths, e.g. [256, 128].
        num_classes:   Number of output classes.
        T_pc:          PC inference steps per sample.
        lr_pc:         Step size for representation updates during PC inference.
        ce_weight:     Cross-entropy loss weight.
        beta:          CV-LIF membrane decay constant.
        threshold:     CV-LIF firing threshold on |U|.
        timesteps:     Number of SNN timesteps for temporal encoding.
        alpha:         Temporal encoding decay constant.
        omega:         Temporal encoding phase factor.
        mode:          'full' | 'pc_only' | 'stdp_only'.
        stdp_eta_r:    STDP magnitude learning rate.
        stdp_eta_th:   STDP phase learning rate.
        stdp_lam:      Phase coherence regulariser strength.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
        alpha: float = 3.0,
        omega: float = 1.0,
        mode: str = "full",
        stdp_eta_r: float = 1e-4,
        stdp_eta_th: float = 1e-4,
        stdp_lam: float = 0.01,
    ) -> None:
        super().__init__()

        assert mode in ("full", "pc_only", "stdp_only"), (
            f"mode must be 'full', 'pc_only', or 'stdp_only', got {mode!r}"
        )

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight
        self.timesteps = timesteps
        self.alpha = alpha
        self.omega = omega
        self.mode = mode
        self._n_hidden = len(hidden_dims)

        # CV-LIF layers for the bottom-up spiking pass
        dims = [input_size] + hidden_dims + [num_classes]
        self.cv_lif_layers = nn.ModuleList([
            ComplexLIFLayer(dims[i], dims[i + 1], beta=beta, threshold=threshold)
            for i in range(len(dims) - 1)
        ])

        # Separate generative (PC) weights: map from upper layer → lower layer prediction
        # W_pc^(l): d_{l+1} → d_l  (predictive direction, top-down)
        self.pc_layers = nn.ModuleList([
            ComplexLinear(dims[i + 1], dims[i], bias=True)
            for i in range(len(dims) - 1)
        ])

        # Classification head on real-valued spike magnitudes of r^(L-1)
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)

        # Complex STDP updater (applied to pc_layers)
        self.stdp = ComplexSTDPUpdater(
            eta_r=stdp_eta_r,
            eta_th=stdp_eta_th,
            lam=stdp_lam,
        )

    # ------------------------------------------------------------------
    # Bottom-up spiking pass
    # ------------------------------------------------------------------

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """Run CV-LIF bottom-up pass.

        Converts real input to complex temporal encoding, then runs
        self.timesteps of CV-LIF dynamics through each layer.

        Returns:
            List [r^(0), r^(1), ..., r^(L)] of complex representations,
            shape (N, d_l) each. r^(0) = complex-encoded input.
        """
        device = x_flat.device
        batch_size = x_flat.shape[0]

        # Encode input as complex temporal representation (averaged over timesteps)
        r_input_real = x_flat.clamp(0.0, 1.0)
        r_input = real_to_complex_temporal(r_input_real, self.alpha, self.omega)  # (N, 784), cfloat

        reps: List[torch.Tensor] = [r_input.detach()]

        current = r_input.unsqueeze(0).expand(self.timesteps, -1, -1)  # (T, N, d)

        for layer_idx, cv_lif in enumerate(self.cv_lif_layers):
            mem = cv_lif.init_mem(batch_size, device)
            spk_acc = torch.zeros(batch_size, cv_lif.fc.out_features, dtype=torch.cfloat, device=device)
            spk_rec = []
            for t in range(self.timesteps):
                spk, mem = cv_lif(current[t], mem)
                spk_acc = spk_acc + spk
                spk_rec.append(spk)

            # Average spike representation over timesteps (complex mean)
            r_l = (spk_acc / self.timesteps).detach()
            reps.append(r_l)

            # Feed complex spikes to next layer
            current = torch.stack(spk_rec)  # (T, N, out)

        return reps

    # ------------------------------------------------------------------
    # PC energy helpers
    # ------------------------------------------------------------------

    def _pc_energy(
        self,
        reps: List[torch.Tensor],
        detach_weights: bool,
    ) -> torch.Tensor:
        """Total PC free energy F = Σ_l ‖r^(l) - W_pc^(l) · r^(l+1)‖²."""
        energy = reps[0].new_zeros((), dtype=torch.float32)
        for l in range(len(self.pc_layers)):
            energy = energy + _complex_energy(
                reps[l], reps[l + 1], self.pc_layers[l], detach_weights=detach_weights
            )
        return energy

    def _run_inference(
        self,
        reps: List[torch.Tensor],
        clamp_last: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """PC inference: T_pc gradient-descent steps on representations.

        Args:
            reps:       Initial representations [r^(0), ..., r^(L)].
            clamp_last: If given (one-hot targets as real), clamp r^(L) to this.

        Returns:
            Updated representations with r^(0) fixed (detached input).
        """
        r0 = reps[0].detach()
        # Convert clamp_last to complex if provided (real → complex with zero phase)
        if clamp_last is not None:
            clamp_c = torch.polar(clamp_last, torch.zeros_like(clamp_last)).to(torch.cfloat)
        else:
            clamp_c = None

        r_free = [r.detach().clone() for r in reps[1:]]
        if clamp_c is not None:
            r_free[-1] = clamp_c.detach()

        with torch.enable_grad():
            for r in r_free:
                r.requires_grad_(True)

            for _ in range(self.T_pc):
                update_reps = r_free[:-1] if clamp_c is not None else r_free
                if not update_reps:
                    break
                energy = self._pc_energy([r0] + r_free, detach_weights=True)
                grads = torch.autograd.grad(energy, update_reps, create_graph=False)
                updated = [
                    (r - self.lr_pc * g).detach().requires_grad_(True)
                    for r, g in zip(update_reps, grads)
                ]
                if clamp_c is not None:
                    r_free = updated + [r_free[-1]]
                else:
                    r_free = updated

        return [r0] + [r.detach() for r in r_free]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pc_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training loss.

        Returns:
            combined_loss, energy (detached, for logging), logits (detached).
        """
        x_flat = x.view(x.shape[0], -1)
        target = F.one_hot(y, self.num_classes).float()

        # --- Bottom-up complex spiking pass ---
        reps_init = self._bottom_up(x_flat)

        # --- PC inference (clamped) → PC weight-update energy ---
        reps_final = self._run_inference(reps_init, clamp_last=target)

        if self.mode in ("full", "pc_only"):
            # PC energy: gradient flows through pc_layers weights
            energy = self._pc_energy(reps_final, detach_weights=False)
        else:
            energy = reps_init[0].new_zeros((), dtype=torch.float32)

        # --- CE loss on classification head applied to |r^(L-1)| ---
        # Use magnitude of the PC-inferred hidden representation
        r_hidden = reps_final[-2].abs()   # real-valued magnitudes, (N, d_{L-1})
        logits = self.cls_head(r_hidden)
        ce_loss = F.cross_entropy(logits, y)

        combined_loss = energy + self.ce_weight * ce_loss

        # --- Complex STDP update (applied after gradient step, in training) ---
        # We store the representations for the STDP update; it's called from outside
        # (train_pcnn loop calls optimizer.step() then model.stdp_step())
        # Store internally so stdp_step() has access.
        if self.mode in ("full", "stdp_only"):
            self._reps_for_stdp = reps_final

        return combined_loss, energy.detach(), logits.detach()

    @torch.no_grad()
    def stdp_step(self) -> None:
        """Apply Complex STDP update to pc_layers using cached representations.

        Must be called after optimizer.step() in the training loop.
        Computes prediction errors from the last pc_loss() call's representations.
        """
        if not hasattr(self, "_reps_for_stdp"):
            return
        reps = self._reps_for_stdp

        for l, pc_layer in enumerate(self.pc_layers):
            r_lower = reps[l]       # (N, d_l)
            r_upper = reps[l + 1]   # (N, d_{l+1})

            # Prediction and complex error
            w = pc_layer.weight.detach()
            b = pc_layer.bias.detach() if pc_layer.bias is not None else None
            mu = F.linear(r_upper, w, b)       # (N, d_l)
            eps = r_lower - mu                  # complex error, (N, d_l)

            self.stdp.update(pc_layer, r_lower, r_upper, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation forward pass: SNN bottom-up → free PC inference → cls_head."""
        x_flat = x.view(x.shape[0], -1)
        reps_init = self._bottom_up(x_flat)
        reps_free = self._run_inference(reps_init)
        r_hidden = reps_free[-2].abs()
        return self.cls_head(r_hidden)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def prediction_errors(self, reps: List[torch.Tensor]) -> List[float]:
        """Compute per-layer prediction error norms for logging."""
        errors = []
        for l, pc_layer in enumerate(self.pc_layers):
            w = pc_layer.weight.detach()
            b = pc_layer.bias.detach() if pc_layer.bias is not None else None
            mu = F.linear(reps[l + 1], w, b)
            eps = reps[l] - mu
            errors.append((eps.abs() ** 2).mean().item())
        return errors

    def phase_coherence(self) -> List[float]:
        """Compute per-layer mean phase coherence index of pc_layer weights.

        Returns phase coherence R = |Σ exp(jθ)| / N ∈ [0,1].
        R≈1 means all phases aligned; R≈0 means uniformly dispersed.
        """
        coherences = []
        for pc_layer in self.pc_layers:
            phases = pc_layer.weight.angle()  # (out, in)
            R = torch.exp(1j * phases.to(torch.cfloat)).mean().abs().item()
            coherences.append(R)
        return coherences

    def spike_count(self, reps: List[torch.Tensor]) -> float:
        """Mean spike count per neuron across all hidden representations."""
        counts = [r.abs().mean().item() for r in reps[1:-1]]
        return sum(counts) / max(len(counts), 1)
