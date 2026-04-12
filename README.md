# chpc_autoresearch

Automated AI Research Assistant for the [CHPC](https://www.chpc.ac.za/) (Center for High Performance Computing, South Africa).

An agentic framework where AI coding agents conduct full research loops -- literature survey, implementation, experimentation on HPC, analysis, and documentation -- all driven by structured state files so sessions can pick up where the last left off.

Github URL: git@github.com:Neurulation/CHPC-Autoresearch.git

## How It Works

1. **Human gives direction** -- e.g., "Research NN architectures for image classification"
2. **Agent creates a project** with iterations (sprints)
3. Each iteration follows the **research loop**: survey -> implement -> experiment -> analyze -> conclude
4. **State is tracked** in YAML files so new agent sessions can resume
5. **Experiments run on CHPC** via PBS job scripts (multiple can run concurrently)
6. **Results are analyzed** and the **leaderboard** below is updated

### Starting the Agent

**Claude Code:**
```
/start [YOUR RESEARCH TOPIC]
```

**GitHub Copilot (or any agent):** Just say "start" -- the agent reads `.github/copilot-instructions.md` and knows what to do.

**Manual prompt** (if slash commands aren't available):
> Read CLAUDE.md and run /state-resume. Continue the autoresearch loop from wherever it left off. If no project exists, create one for: **[YOUR RESEARCH TOPIC]**. Work through each phase autonomously. For experiments, try SSH to CHPC directly; if that fails, give me the commands. Commit after each phase.

### Git + CHPC Strategy

- **Development** happens on feature branches, merged to `develop` when ready
- **CHPC always stays on `develop`** -- experiments are distinguished by Hydra configs, not branches
- This means **multiple experiments can run concurrently** on CHPC (each writes to its own output directory)
- The agent implements code -> merges to develop -> you pull on CHPC and `qsub`

## Quick Start

```bash
# Clone and install
git clone <repo_url>
cd chpc_autoresearch
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your CHPC credentials and WANDB key

# Run a quick local test
python -m autoresearch.train experiment=mnist_ffnn_adam seed=0 epochs=2 wandb.enabled=false

# Run full experiment (single seed)
python -m autoresearch.train experiment=mnist_ffnn_adam seed=0

# Run multi-seed sweep
python -m autoresearch.train --multirun experiment=mnist_ffnn_adam
```

## Available Experiments

| Experiment | Dataset | Model | Optimizer | Config |
|-----------|---------|-------|-----------|--------|
| `mnist_ffnn_adam` | MNIST | FFNN (784-256-128-10) | Adam | 50 epochs, 5 seeds |
| `mnist_cnn_adam` | MNIST | CNN (2 conv + 1 FC) | Adam | 50 epochs, 5 seeds |
| `mnist_snn_baseline_adam` | MNIST | SNN (784-512-256-10 LIF, T=25) | Adam | 20 epochs, 5 seeds |
| `cifar10_resnet18_adam` | CIFAR-10 | ResNet-18 (CIFAR-modified) | Adam | 100 epochs, 5 seeds |

## Research Roadmap: ANN → SNN

The overarching goal is a **fully biologically plausible, end-to-end spiking implementation** with three requirements:
1. **No backpropagation** — replaced by local learning rules (Hebbian, STDP) in later phases
2. **Fully spike-driven** — every inter-layer signal is binary spikes {0,1}; no float hidden states between layers
3. **Neuromorphic-deployable** — theoretically runnable on energy-efficient spiking hardware

### Architecture Ladder

| Step | ANN | Fully Spiking SNN | snntorch primitive | Status |
|------|-----|--------------------|-------------------|--------|
| 1 | FFNN | **SFNN** | `snn.Leaky` per FC layer | Done ✓ |
| 2 | CNN | **SCNN** | `snn.Leaky` after each conv | Done ✓ |
| 3 | Vanilla RNN | **SRNN** | `snn.RLeaky(linear_features=N)` — recurrent LIF | Planned |
| 4 | LSTM | **SLSTM** | `snn.SLSTM(input_size, hidden_size)` | Planned |
| 5 | GRU | **SGRU** | No native — custom LIF-gated GRU | Deferred |
| 6 | Transformer | **STransformer** | No native — research-level | Deferred |

**Hybrid ≠ Fully Spiking.** Phase B/C experiments (iters 3–10) used a hybrid architecture:
`snn.Leaky` encoder → spike counts → standard `nn.LSTM/GRU/RNN`. This is NOT the target
fully spiking design. These results answer a separate but useful question: *do spike-encoded
inputs help standard RNNs?* The genuine SRNN and SLSTM implementations are in iters 11–14.

## Leaderboard

Results grouped by dataset. Val Acc = mean ± std across seeds where available. ⚠️ = preliminary (single seed / partial run).

### MNIST

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Notes | Status |
|---------|------|-------|-----------|-----|-------|---------|-------|--------|
| Image Processing NN | 1 | CNN (2 conv + 1 FC) | Adam | ✗ | 5 | **99.17% ± 0.10%** | spatial | ✅ |
| ANP — RNN | 2 | GRU (2-layer, h=256) | Adam | ✗ | 5 | **99.06% ± 0.14%** | sequential (T=28) | ✅ |
| ANP — RNN | 1 | LSTM (2-layer, h=256) | Adam | ✗ | 5 | 98.95% ± 0.11% | sequential (T=28) | ✅ |
| ANP — SNN | 1 | SNN-CNN (2 Conv+LIF+Pool + 2 FC+LIF, T=25) | Adam | ✗ | 5 | **98.87% ± 0.13%** | rate-coded, spatial | ✅ |
| ANP — SNN | 4 | **Hybrid**-GRU (LIF encoder + 2-layer GRU h=256, T=25) | Adam | ✗ | 5 | 98.75% ± 0.15%ᴮ | hybrid baseline — not fully spiking | ✅ |
| ANP — SNN | 3 | **Hybrid**-LSTM (LIF encoder + 2-layer LSTM h=256, T=25) | Adam | ✗ | 5 | 98.68% ± 0.04%ᴮ | hybrid baseline — not fully spiking | ✅ |
| ANP — SNN | 7 | SCNN TTFS (2 Conv+LIF+Pool + 2 FC+LIF, TTFS T=25) | Adam | ✗ | 5 | 98.41% ± 0.17%ᶜ | fully spiking, temporal coding | ✅ |
| Image Processing NN | 1 | FFNN (784-256-128-10) | Adam | ✗ | 5 | 98.06% ± 0.15% | dense | ✅ |
| ANP — RNN | 3 | Vanilla RNN (2-layer, h=256) | Adam | ✗ | 5 | 97.89% ± 0.35% | sequential (T=28) | ✅ |
| ANP — SNN | 2 | SNN-FFNN (784-512-256-10 LIF, T=25) | Adam | ✗ | 5 | 97.62% ± 0.12%ᴬ | rate-coded (Phase B baseline) | ✅ |
| ANP — PC-NN | 3 | PC-FFNN v3 + CE head + grad clip (784-256-128-10) | Adam | ✗ | 5 | **97.30% ± 0.22%**⁴ | predictive coding | ✅ |
| ANP — PC-NN | 2 | PC-FFNN v2 + CE head (784-256-128-10) | Adam | ✗ | 5 | **97.21% ± 0.35%**³ | predictive coding | ✅ |
| ANP — PC-NN | 10 | PC-EncDec v2 + cosine LR v2 (784-256-128 enc+dec, β=0.1) | Adam+Cosine | ✗ | 5 | 96.75% ± 0.19%¹⁰ | cosine LR degraded −0.39pp vs flat LR | ✅ |
| ANP — PC-NN | 9 | PC-EncDec v2 + cosine LR (784-256-128 enc+dec, β=0.1) | Adam+Cosine | ✗ | 5 | ~~95.82% ± 0.20%~~⁹ (INVALID) | wrong entry point + ES bug | ❌ |
| ANP — PC-NN | 8 | PC-EncDec v2 @ 60ep (784-256-128 enc+dec, β=0.1) | Adam | ✗ | 5 | **97.14% ± 0.21%**⁸ | generative PC, 60ep ceiling | ✅ |
| ANP — SNN | 6 | SFNN TTFS (784-512-256-10 LIF, TTFS T=25) | Adam | ✗ | 5 | 97.12% ± 0.20%ᶜ | fully spiking, temporal coding | ✅ |
| ANP — PC-NN | 7 | PC-EncDec v2 (784-256-128 enc+dec, β=0.1) | Adam | ✗ | 5 | **96.59% ± 0.28%**⁷ | generative PC, ff-forward eval | ✅ |
| ANP — PC-NN | 1 | PC-FFNN (784-256-128-10) | Adam | ✗ | 5 | ~96%¹ / 89.0% ± 4.7%² | predictive coding | ⚠️ |
| ANP — PC-NN | 4 | PC-FFNN v4 + eps=0.01 (784-256-128-10) | Adam (eps=0.01) | ✗ | 5 | 93.95% ± 0.27%⁵ | not converged | ⚠️ |
| ANP — PC-NN | 6 | PC-EncDec v1 (784-256-128 enc+dec) | Adam | ✗ | 5 | 93.13% ± 0.35%⁶ | train/val mismatch + β=1 | ⚠️ |
| ANP — SNN | 5 | **Hybrid**-VanillaRNN (LIF encoder + 2-layer RNN h=256, T=25) | Adam | ✗ | 5 | 97.17% ± 0.52%ᴮ | hybrid baseline — not fully spiking | ✅ |
| ANP — SNN | 8 | **Hybrid**-LSTM TTFS (LIF encoder + LSTM, TTFS T=25) | Adam | ✗ | 5 | ⚠️ 2/5 seedsᶜ | walltime kill — resubmitting 4h | ⚠️ |
| ANP — SNN | 9 | **Hybrid**-GRU TTFS (LIF encoder + GRU, TTFS T=25) | Adam | ✗ | 5 | ⚠️ 2/5 seedsᶜ | walltime kill — resubmitting 4h | ⚠️ |
| ANP — SNN | 10 | **Hybrid**-VanillaRNN TTFS (LIF encoder + RNN, TTFS T=25) | Adam | ✗ | 5 | ⚠️ 2/5 seedsᶜ | walltime kill — resubmitting 4h | ⚠️ |
| ANP — SNN | 11 | **SRNN** rate (snn.RLeaky T=28, h=256) | Adam | ✗ | 5 | plannedᴰ | **fully spiking recurrent** | 📋 |
| ANP — SNN | 12 | **SLSTM** rate (snn.SLSTM T=28, h=256) | Adam | ✗ | 5 | plannedᴰ | **fully spiking recurrent** | 📋 |
| ANP — SNN | 13 | **SRNN** TTFS (snn.RLeaky T=28, TTFS) | Adam | ✗ | 5 | plannedᴰ | **fully spiking recurrent** | 📋 |
| ANP — SNN | 14 | **SLSTM** TTFS (snn.SLSTM T=28, TTFS) | Adam | ✗ | 5 | plannedᴰ | **fully spiking recurrent** | 📋 |
| ANP — SPCNN | 2 | SPC-FFNN v1 (SNN-FFNN + PC inference loop) | Adam | ✗ | 5 | ~11%ˢ | COMPLETE FAILURE | ❌ |
| ANP — SPCNN | 3b | SPC-FFNN v2 (PC energy + CE on detached SNN reps) | Adam | ✗ | 5 | ~11%ˢ | COMPLETE FAILURE — shared weights | ❌ |
| ANP — SPCNN | 4 | SPC-FFNN A (CE through LIF surrogate grads) | Adam | ✗ | 5 | ~11%ˢ | COMPLETE FAILURE — competing gradients | ❌ |
| ANP — SPCNN | 5 | SPC-FFNN D (CE via BPTT through PC inference) | Adam | ✗ | 5 | ~11%ˢ | COMPLETE FAILURE — uniform collapse | ❌ |

ᴬ **ANP — SNN iter 2 (SNN-FFNN rate coding):** 97.62% ± 0.12% (535k params). Rate coding (T=25 Bernoulli) preserves
  accuracy well — only −0.44pp gap vs non-spiking FFNN despite larger hidden dims (512-256 vs 256-128).  
ᴮ **ANP — SNN iters 3-5 (Phase B: Hybrid-LSTM/GRU/VanillaRNN — complete):** Jobs 7171472-7171474.
  Architecture: `snn.Leaky` input encoder (T=25 Bernoulli per row) → rate-coded spike counts → standard `nn.LSTM/GRU/RNN`.
  **These are hybrid comparison baselines, NOT the target fully spiking design.**
  They answer: "do spike-encoded inputs help standard RNNs?"
  Results (all 3 complete, 5/5 seeds each, 30 epochs):
    SNN-GRU (iter4): **98.75% ± 0.15%** — gap vs non-spiking GRU: −0.31pp
    SNN-LSTM (iter3): **98.68% ± 0.04%** — gap vs non-spiking LSTM: −0.27pp
    SNN-VanillaRNN (iter5): **97.17% ± 0.52%** — gap vs non-spiking VanillaRNN: −0.72pp
  Rate-coded spiking input is near-lossless for gated models (−0.27–0.31pp). VanillaRNN gap (−0.72pp)
  is 2.3× larger — gating absorbs spike encoding loss; ungated RNNs are more sensitive.
  High variance for VanillaRNN (0.52pp std vs 0.04-0.15pp) amplified by vanishing gradients + sparse spikes.
  **Phase B conclusion: gated architectures (LSTM/GRU) absorb spike encoding loss; plain RNNs do not.**  
ᶜ **ANP — SNN iters 6-10 (Phase C TTFS):** iters 6-7 (SFNN/SCNN TTFS) fully spiking — complete.
  iters 8-10 (LSTM/GRU/VanillaRNN TTFS) are hybrid baselines.
  Tests whether TTFS temporal coding improves over rate coding for static/sequential MNIST.
  Results (iters 6/7 complete): SFNN TTFS **97.12% ± 0.20%**, SCNN TTFS **98.41% ± 0.17%**.
  TTFS is consistently WORSE than rate: −0.49pp (FFNN) and −0.46pp (CNN). Root cause: LIF summation
  discards spike timing order — TTFS ≈ binarized input; rate coding preserves graded intensity.
  iter8b/9b/10b (corrected threshold=0.9 reruns, jobs 7172625/6/7): walltime-killed at 2/5 seeds
  (~1h/seed × 5 = 5h needed; 2h PBS insufficient). Resubmitting with walltime=04:00:00.
  Partial iter8b signal (2 seeds): LSTM TTFS seed0 ~98.79% > LSTM rate 98.68% — weak early indicator
  TTFS may benefit recurrent models; too few seeds to conclude.  
ᴰ **ANP — SNN iters 11-14 (fully spiking recurrent — planned):** Next priority after iters 3-10 complete.
  SRNN uses `snn.RLeaky(linear_features=256)` over T=28 MNIST rows — fully binary spikes throughout.
  SLSTM uses `snn.SLSTM(28, 256)` — standard LSTM gates internally but thresholded membrane output → binary spikes.
  These are the first *truly* fully spiking recurrent models in the ladder; no standard PyTorch RNN cells.  
² Mean last-epoch val_acc at early-stop (epochs 9-11). High variance and degradation compared to best epoch
  is caused by a training-evaluation objective mismatch (see key findings below).  
³ Best-epoch val_acc across 5 seeds (early stopping on val_acc, mode=max). CE head resolves iter 1 calibration failure.
  A new issue emerged: PC energy explosions mid-training (all seeds); early stopping preserves the best model correctly.  
⁴ Grad clipping (max_grad_norm=0.5) delays explosions and reduces variance (std 0.35→0.22) but does not eliminate them.
  Explosions are algorithmic — the CE head and PC energy compete for the same weights. Iter 4: reduce ce_weight=0.1.  
⁵ **Stability result, not a performance result.** Adam eps=0.01 eliminates energy explosions (zero across 5 seeds) at the
  cost of slower convergence. 30 epochs insufficient; single-seed diagnostic at 60+ epochs reached 98.12%. Needs ~50-75
  epochs to show true capability. Future re-run with epochs=75 will establish PC-FFNN ceiling.  
⁶ **Train/val mismatch + Y_max too high.** Energy schedule fix resolved the original <1% CE-gradient problem (prev: ~31%).
  93.13% ceiling caused by: (1) cls_head trained on feedforward r_2 but validated on inference-modified r_2 (mismatch);
  (2) Y_max=0.5 = β=1 VAE — reconstruction and classification compete equally, suboptimal for discrimination.
  Iter 7 fixes: pure-feedforward forward() + Y_max=0.1 (β=0.1).  
⁷ **Both fixes confirmed.** +3.46pp vs iter 6 (96.59% vs 93.13%). Fix 1 (train/val mismatch closure) was the dominant
  contributor. Fix 2 (β=1→β=0.1, 50/50 → 90/10 CE/energy gradient split) added secondary improvement.
  All seeds best at epochs 26-30 — model still improving at epoch 30. 30 more epochs may yield further gains.
  0.71pp below PC-FFNN v3 (97.30%); the generative decoder is now a mild regulariser, not a liability.
⁸ **Training budget closes the efficiency gap.** +0.55pp over 30ep (96.59%→97.14%). Seeds 1 and 2 early-stopped
  (patience=10), seeds 0/3/4 needed all 60 epochs (slow convergence). Gap to PC-FFNN v3 reduced from -0.71pp to -0.16pp.
  Architecture is competitive; bottleneck is slow convergence. Cosine LR decay recommended for iter 9.
⁹ **anp_pcnn iter 9 (PC-EncDec v2 + cosine LR — INVALID):** Job 7171630, 95.82%±0.20% at epoch 11.
  Two bugs invalidated the result: (1) PBS script used `autoresearch.train` instead of `autoresearch.train_pcnn`
  (wrong entry point); (2) `train.py` evaluates `val_loss` with `mode='max'` → early stopping counter increments
  every epoch with improving val_loss → fires at patience+1 = 11 epochs for all 5 seeds deterministically.
  Cosine LR was applied (train.py has scheduler support) but unmeasurable — LR barely changed by epoch 11.
  Fixes applied to `train_pcnn.py`: early stopping reverted to `val_accuracy` (mode=max); scheduler support added.
¹⁰ **anp_pcnn iter 10 (PC-EncDec v2 + cosine LR — valid rerun, complete):** Job 7172728, 5/5 seeds.
  96.75% ± 0.19% (epochs [60, 56, 53, 55, 60]). Seeds 1/2/3 early-stopped (patience=10 on val_accuracy).
  **Cosine LR DEGRADED performance −0.39pp vs flat LR iter8 (97.14%).**
  PC models are LR-sensitive: aggressive cosine decay over 60 epochs stalls the slow inference-phase
  credit assignment before convergence. Flat LR (iter8) remains the best PC-EncDec result.
  Still −0.55pp behind PC-FFNN v3 (97.30%). Cosine annealing is not beneficial for PC-EncDec on MNIST.
ˢ **SPC-FFNN architectural failure (iters 2-5, anp_spcnn) — ALL variants produce chance accuracy (~11%):**
  Root cause: `self.layers` is shared between the SNN feedforward pathway (binary spike processing for classification)
  and the PC generative model (continuous representation reconstruction). These objectives are architecturally incompatible:
  - **v2 (iter3b):** SNN outputs detached — CE only updates `cls_head`; PC energy trains `self.layers` for reconstruction.
    Reconstructive objective alone cannot produce discriminative features → chance.
  - **Option A (iter4):** CE flows through LIF surrogate gradients to `self.layers`. Competing CE + PC energy gradients
    cause energy explosion (7.9→57.6 over 8 epochs); optimization collapses → chance.
  - **Option D (iter5):** CE via BPTT through T_pc=10 PC inference steps. Two-pass overhead adds no benefit over A;
    most uniform failure (val_loss 2.3021-2.3023 all seeds) → complete uniform prediction collapse.
  **Required fix (iter6+):** Separate SNN encoder weights from PC generative model weights. SNN encoder trained
  discriminatively (CE + surrogate grads). PC generative model has its own separate weight matrices.
  No architecture with shared SNN/PC weights can reconcile these gradient conflicts.
  Same bug class as PC-EncDec v1 (anp_pcnn iter 6). Fix (iter 3): CE now uses `reps_init[-2]` from `_bottom_up()`;
  `forward()` changed to pure-feedforward SNN (no PC inference). Smoke test: 39.43%→46.72% @ ep1-2.
  Job 7171612, queued 2026-04-12.

### CIFAR-10

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Status |
|---------|------|-------|-----------|-----|-------|---------|--------|
| Image Processing NN | 3 | ResNet-18 | SGD+Cosine | ✓ | 5 | **94.96% ± 0.38%** | ✅ |
| Image Processing NN | 3 | ResNet-18 | Adam | ✓ | 5 | 90.57% ± 0.51% | ✅ |
| Image Processing NN | 1 | ResNet-18 | Adam | ✗ | 5 | 83.56% ± 0.36% | ✅ |
| Image Processing NN | 2 | ResNet-18 | SGD+Cosine | ✗ | 5 | 78.87% ± 0.94% | ✅ |

*Updated 2026-04-12.*

**Key findings:**
- *MNIST: CNN outperforms FFNN at 99.17% vs 98.06%. SNN-CNN (iter1, anp_snn) 98.87% ± 0.13% — only -0.30pp behind non-spiking CNN, confirming LIF neurons + rate coding are effective for spatial feature extraction. SNN-FFNN baseline (iter2, anp_snn) 97.62% ± 0.12% — -0.44pp vs non-spiking FFNN despite larger hidden dims (512-256 vs 256-128). Architecture gain: +1.25pp from adding convolutional spiking layers.*
- *SNN Phase B (iters 3-5, anp_snn) — hybrid spiking recurrent results (all complete): SNN-GRU 98.75% ± 0.15% (iter4), SNN-LSTM 98.68% ± 0.04% (iter3), SNN-VanillaRNN 97.17% ± 0.52% (iter5). Rate-coded spiking input is near-lossless for gated recurrent models: −0.27pp (LSTM) and −0.31pp (GRU) gap vs non-spiking counterparts. VanillaRNN gap −0.72pp is 2.3× larger — demonstrates that gating (LSTM/GRU gates filter + compress input signal) absorbs spike encoding loss; plain RNN without gating is more sensitive to input precision. High variance for VanillaRNN (0.52pp std vs 0.04-0.15pp for LSTM/GRU) amplified by interaction of vanishing gradients with sparse spike inputs.*
- *SNN Phase C TTFS (iters 6-7, anp_snn) — TTFS is consistently WORSE than rate coding for static MNIST: SFNN TTFS 97.12% ± 0.20% vs rate 97.62% (−0.49pp, iter6); SCNN TTFS 98.41% ± 0.17% vs rate 98.87% (−0.46pp, iter7). Remarkably consistent ~0.47pp penalty across two different architectures points to the encoding itself, not the downstream model, as the cause. Mechanism: LIF spike-count integration discards temporal order information — TTFS spike-count = binarized image (pixel fires or not within T steps), whereas rate coding preserves graded intensity via Bernoulli sampling. For static MNIST, graded intensity > spike timing.*
- *GRU (iter 2) beats LSTM (iter 1): 99.06% ± 0.14% vs 98.95% ± 0.11%, with 25% fewer parameters (617k vs ~821k). Gate reduction (4→3 gates) did not hurt — confirms GRU parity with LSTM on seq-MNIST (Chung et al. 2014).*
- *Vanilla RNN (iter 3): 97.89% ± 0.35% — far better than predicted. Literature expects 10-20pp regression from LSTM for T>>10 (Bengio et al. 1994); actual gap from GRU is only 1.17pp. Adam's adaptive LR compensates for vanishing gradients at T=28, acting as a significant equaliser. Completes the RNN trilogy: GRU (99.06%) → LSTM (98.95%) → Vanilla (97.89%). Parameter efficiency: 207k vs 617k (GRU) for 1.17pp.*
- *LSTM/GRU on sequential MNIST (T=28): competitive with CNN despite processing pixels row-by-row.*
- *PC-FFNN v4 (iter 4): eps=0.01 fix CONFIRMED zero energy explosions across all 5 seeds (energy monotonically decreases to ~0.21 at ep30). However 93.95% is a convergence artifact — not a performance comparison. Adam eps=0.01 reduces effective step size in late training, needing ~50-75 epochs to match the single-seed diagnostic of 98.12%. A future re-run with epochs=75 will establish the PC-FFNN ceiling.*
- *PC-FFNN v3 (iter 3): Gradient clipping (max_grad_norm=0.5) is a partial improvement — variance reduced (0.35→0.22pp), explosions delayed, mean accuracy +0.09pp to 97.30% ± 0.22%. Root cause: clipping bounds gradient magnitude but not the energy value itself.*
- *PC-FFNN v1 (iter 1): train accuracy 100% from epoch 2 via supervised clamping, but val CE stuck at ~1.54 (uncalibrated). Root cause: training-evaluation objective mismatch between clamped and free inference.*
- *PC-EncDec v2 @ 60ep (iter 8): 97.14% ± 0.21% — +0.55pp over 30ep (96.59%). Training budget alone closed 78% of the gap to PC-FFNN v3 (-0.71pp→-0.16pp). Seeds 1/2 early-stopped; seeds 0/3/4 needed all 60 epochs — slow convergence is the main bottleneck. Cosine LR decay tested in iter 10 — result: DEGRADED by −0.39pp (96.75%). Flat LR remains optimal for PC-EncDec.*
- *PC-EncDec v2 + cosine LR (iter 10): 96.75% ± 0.19% — cosine LR (T_max=60, eta_min=1e-6) HURT convergence by −0.39pp vs flat LR iter8 (97.14%). PC models are more LR-sensitive than backprop models: aggressive decay stalls slow inference-phase credit assignment before full convergence. Seeds 1/2/3 early-stopped (patience=10 on val_accuracy) while seed 0/4 ran full 60ep. Flat LR (iter8) remains best PC-EncDec result. Still −0.55pp behind PC-FFNN v3 (97.30% flat LR). Next direction: longer flat-LR training (120ep) or architectural improvements.*
- *PC-EncDec v2 (iter 7): 96.59% ± 0.28% — +3.46pp vs iter 6 (93.13%). Both fixes confirmed: (1) closing the train/val distribution mismatch (pure-feedforward eval) was the dominant contributor; (2) reducing β from 1.0 to 0.1 (Y_max 0.5→0.1) shifted gradient budget to 90% CE / 10% energy. Generative decoder is now a mild regulariser, not a hindrance. All seeds best at epochs 26-30 — model not yet converged at epoch 30; iter 8 recommended at 60 epochs to establish ceiling.*
- *PC-EncDec (iter 6): 93.13% ± 0.35% ceiling caused by two compounding bugs: (1) train/val mismatch — cls_head trained on feedforward r_{L-1} but validated on inference-modified r_{L-1} (20 PC steps shift the representation distribution); (2) Y_max=0.5 = β=1 VAE — reconstruction and classification compete with equal gradient budget, known suboptimal for discrimination (Higgins et al. 2017). Iter 7 fixes both: pure-feedforward forward() + Y_max=0.1 (β=0.1, 90% CE gradient).*
- ***SPC-FFNN architectural failure (anp_spcnn iters 2-5) — ALL variants COMPLETE FAILURE:** ~11% (chance) across all variants and seeds. Architectural root cause: `self.layers` shared between SNN feedforward pathway and PC generative model. Three gradient routing strategies all fail: (v2, iter3b) PC energy alone — cannot produce discriminative features; (A, iter4) CE through LIF surrogate grads — energy explodes (7.9→57.6), optimization collapses; (D, iter5) CE via BPTT through PC inference — most uniform collapse (val_loss 2.3021-2.3023). Required fix: separate SNN encoder weights from PC generative model weights. SNN encoder trained discriminatively (CE+surrogate); PC model has independent weight matrices. Iter6 will implement this two-pathway architecture.*
- *CIFAR-10: Data augmentation was THE limiting factor. SGD+cosine with aug: 94.96% (+16.09%). Adam with aug: 90.57% (+7.01%). SGD+cosine beats Adam when both use augmentation.*

## CHPC Usage

```bash
# Generate a PBS script
python scripts/generate_pbs.py \
  --name train_mnist_ffnn \
  --commands "python -m autoresearch.train --multirun experiment=mnist_ffnn_adam"

# SSH to CHPC and submit
ssh $CHPC_USERNAME@lengau.chpc.ac.za
cd lustre/chpc_autoresearch
qsub experiments/train_mnist_ffnn.pbs

# Check job status
qstat -u $CHPC_USERNAME
```

### Autonomous Job Monitoring (`/loop`)

After submitting jobs, use `/loop` to have the agent poll CHPC and auto-process results without you having to prompt it:

```
/loop 10m Check CHPC job status. For each completed job, extract results,
write metrics.json, update state, commit+push. Resubmit any killed jobs.
```

**How it works:**
1. `/loop` parses the interval (`10m`) into a standard 5-field cron expression (`*/10 * * * *`)
2. It calls `CronCreate` with that expression and your prompt — `CronCreate` returns a job ID (e.g. `eb3ec6ae`)
3. The prompt runs **immediately**, then repeats on schedule
4. Each firing only happens while Claude Code is **idle** (never interrupts mid-query)
5. The job is **session-only** — lost when Claude Code closes — and auto-expires after 7 days

Cancel any time with the job ID printed at scheduling:

```
CronDelete("eb3ec6ae")
```

Or just tell the agent: *"stop the loop"* / *"cancel monitoring"*.

**Walltime sizing rule:** `walltime = n_seeds × per_seed_time × 1.2` (20% buffer).
RNN-class models: use 4h. FFNN/CNN: use 2h. GPU-1 queue max is 48h.

**Walltime kill + resume:** `last_checkpoint.pt` is saved every epoch. Resubmitting
the same PBS script is safe — seeds that already finished skip immediately; killed
seeds resume from the last checkpoint. No wasted compute.

See [docs/chpc/](docs/chpc/) for detailed CHPC documentation.

## Project Structure

```
src/autoresearch/       # Main Python package (Hydra + PyTorch)
  train.py              # Training entry point
  configs/              # Hydra YAML configs (dataset, model, optimizer, experiment)
  models/               # FFNN, CNN, ResNet18
  utils/                # Data loading, evaluation, WANDB, reproducibility

projects/               # Research state tracking (YAML)
templates/              # PBS script templates
scripts/                # Helper scripts (PBS generator)
docs/                   # Architecture docs, CHPC guides
.claude/commands/       # Agent skills (slash commands)
```

## Agent Skills

| Skill | Purpose |
|-------|---------|
| `/snntorch-docs` | Look up snntorch neuron classes, equations, and API from local docs |
| `/start` | **Kickoff / resume the full autoresearch loop** |
| `/loop` | **Schedule a recurring poll** — monitors CHPC jobs, auto-analyzes results |
| `/chpc-submit` | Generate PBS script and submit to CHPC (tries SSH directly) |
| `/chpc-status` | Check CHPC job status |
| `/chpc-setup` | Set up repo on CHPC for first time |
| `/experiment-run` | Run experiment locally or on CHPC |
| `/experiment-analyse` | Analyse experiment results |
| `/project-init` | Create a new research project |
| `/iteration-init` | Start a new iteration |
| `/state-resume` | Read state and determine next steps |
| `/docs-fetch` | Fetch CHPC wiki docs for offline reference |

## For Your Own Research

1. Fork this repo
2. Configure `.env` with your CHPC credentials
3. Use `/project-init` to create a research project
4. Add models to `src/autoresearch/models/` with matching configs
5. Run experiments locally or on CHPC
6. The agent tracks state so you can iterate continuously

## Tech Stack

- **PyTorch** + **torchvision** -- ML framework
- **Hydra** + **OmegaConf** -- Config-driven experimentation
- **Weights & Biases** -- Experiment tracking
- **PBS/Torque** -- HPC job scheduling (CHPC)

## License

MIT
