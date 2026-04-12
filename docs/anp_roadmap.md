# Artificial Neural Prostheses — Master Research Roadmap

## Vision

Build an **Artificial Neural Prosthesis (ANP)**: a small, biologically plausible auxiliary
network that interfaces with a larger pre-trained source network via a bandwidth-limited
population-activity channel (~128 input channels), predicts the source network's
next-timestep neural activity, and ultimately intervenes to correct anomalous activity
patterns — analogous to a neuroprosthetic device interfacing with biological tissue.

**The ANP is a Spiking Predictive Coding Network (SPCNN)** with an encoder-decoder
structure. Both error units and representation units are spiking (LIF). The enc-dec
topology closes the loop between bottom-up prediction errors and top-down generative
predictions, mirroring the hierarchical PC principle but with spike-timing dynamics.

---

## Dependency Tree

```
[ULTIMATE GOAL]
ANP: SPCNN predicts + corrects next-frame population activity
            │
     ┌──────┴──────┐
     ▼             ▼
[Architecture]  [Training data]
 SPCNN enc-dec   Event-driven temporal sequences
     │             │
  ┌──┴──┐     ┌────┼────┬──────────┐
  ▼     ▼     ▼    ▼    ▼          ▼
[SNN] [PC-  MNIST NMNIST Artif.  Game of
 enc  EncDec Video  DVS   EEG     Life
 dec] ✅(rate)            ←iter2  ←proposed
  │
  ├── LIF neurons ✅ (snn_baseline)
  └── PC enc-dec  ✅ (pc_enc_dec_v2)

         [Encoding scheme comparison]
         Rate coding  vs  Temporal coding (TTFS)
         (what we have)    (what the ANP needs)
```

---

## Phase Overview

| Phase | Scope | Projects | Status |
|-------|-------|----------|--------|
| **A** | Rate-coded baselines — static tasks | `image_processing_nn`, `anp_rnn`, `anp_pcnn` | Mostly ✅ |
| **B** | Spiking baselines — rate coding | `artificial_neural_prostheses`, `anp_snn` | Partial ✅ |
| **C** | Spiking baselines — temporal coding (TTFS) | `anp_snn` | ❌ |
| **D** | Temporal dataset infrastructure | `anp_datasets` | ❌ |
| **E** | Spiking PC (SPCNN) — rate then temporal | `anp_spcnn` | ❌ |
| **F** | ANP assembly — population interface + prediction | `artificial_neural_prostheses` | ❌ |
| **G** | ANP intervention — anomaly correction | `artificial_neural_prostheses` | ❌ |

---

## Phase A — Rate-coded Baselines (Static Tasks)

Establish performance ceilings for every architecture family on MNIST and CIFAR-10.
These are the comparison baselines for all spiking and PC variants.

### A1 — Dense / CNN (`image_processing_nn`)

| Iter | Model | Dataset | Result | Status |
|------|-------|---------|--------|--------|
| 1 | FFNN (784-256-128-10) | MNIST spatial | 98.06% ± 0.15% | ✅ |
| 1 | CNN (2 conv + FC) | MNIST spatial | 99.17% ± 0.10% | ✅ |
| 1–3 | ResNet-18 | CIFAR-10 | 94.96% ± 0.38% (best) | ✅ |

### A2 — Recurrent (`anp_rnn`)

| Iter | Model | Dataset | Result | Status |
|------|-------|---------|--------|--------|
| 1 | LSTM (2L, h=256) | MNIST sequential (T=28) | 98.95% ± 0.11% | ✅ |
| 2 | GRU (2L, h=256) | MNIST sequential (T=28) | 99.06% ± 0.14% | ✅ |
| 3 | Vanilla RNN (2L, h=256) | MNIST sequential (T=28) | 97.89% ± 0.35% | ✅ |

RNN trilogy complete. No further iterations planned.

### A3 — Predictive Coding (`anp_pcnn`)

| Iter | Model | Dataset | Result | Status |
|------|-------|---------|--------|--------|
| 1–3 | PC-FFNN v1/v2/v3 | MNIST spatial | 97.30% ± 0.22% (best) | ✅ |
| 4 | PC-FFNN v4 (eps=0.01) | MNIST spatial | 93.95% ± 0.27% (not converged) | ⚠️ |
| 5 | PC-RNN | MNIST sequential | ~10% (failed — random) | ❌ needs conclude |
| 6–7 | PC-EncDec v1/v2 | MNIST spatial | 96.59% ± 0.28% (30 ep) | ✅ |
| 8 | PC-EncDec v2 (60 ep) | MNIST spatial | ⏳ running (job 7171099) | ⏳ |

**Note on PC-RNN failure (iter 5):** Greedy per-timestep PC inference with BPTT
produced ~10% accuracy (random) across all 5 seeds. Root cause not yet diagnosed —
likely the combination of per-timestep PC gradient steps and BPTT through recurrent
weights creates conflicting update signals. Needs analysis and a fix before spiking
PC-RNN is attempted.

---

## Phase B — Spiking Baselines, Rate Coding

Establish spiking (LIF, snntorch) equivalents of every rate-coded architecture.
All use rate coding (spike count over window T) for direct comparison with Phase A.

### B1 — Spiking Dense / CNN (`artificial_neural_prostheses` + `anp_snn`)

| Model | Dataset | Notes | Status |
|-------|---------|-------|--------|
| SNN-FFNN (784-512-256-10, LIF, T=25) | MNIST | 97.62% ± 0.12% | ✅ |
| SNN-CNN (conv LIF + FC LIF) | MNIST | Direct spiking analogue of CNN | ❌ |

### B2 — Spiking Recurrent (`anp_snn`)

| Model | Dataset | Notes | Status |
|-------|---------|-------|--------|
| SNN-LSTM (LIF gating) | MNIST sequential | snntorch spiking LSTM | ❌ |
| SNN-GRU (LIF gating) | MNIST sequential | snntorch spiking GRU | ❌ |
| SNN-VanillaRNN (LIF) | MNIST sequential | Simplest spiking recurrent | ❌ |

---

## Phase C — Spiking Baselines, Temporal Coding (TTFS)

**Core research question:** Does temporal coding (precise spike timing) outperform
rate coding on sequence prediction tasks? Hypothesis: rate coding performs comparably
on static classification but temporal coding dominates on temporal prediction —
the crossover is the scientific contribution.

Temporal coding scheme: **Time-to-First-Spike (TTFS)** — earlier spike = stronger
activation. Loss: spike-time loss (van Rullen & Thorpe 2001) or surrogate TTFS
gradient (Goltz et al. 2021, snntorch support).

### C1 — Temporal Spiking Dense / CNN (`anp_snn`)

| Model | Encoding | Dataset | Status |
|-------|----------|---------|--------|
| SNN-FFNN | TTFS | MNIST | ❌ |
| SNN-CNN | TTFS | MNIST | ❌ |

### C2 — Temporal Spiking Recurrent (`anp_snn`)

| Model | Encoding | Dataset | Status |
|-------|----------|---------|--------|
| SNN-LSTM | TTFS | MNIST sequential | ❌ |
| SNN-GRU | TTFS | MNIST sequential | ❌ |
| SNN-VanillaRNN | TTFS | MNIST sequential | ❌ |

### C3 — Rate vs Temporal Comparison Summary

Cross-architecture comparison table produced at end of Phase C.
Primary metric: val accuracy on static MNIST + sequential MNIST.
Hypothesis test repeated on temporal datasets (Phase D/E).

---

## Phase D — Temporal Dataset Infrastructure (`anp_datasets`)

All Phase E–G experiments require event-driven, temporal data. This phase
builds and validates every dataset the ANP training pipeline will use.

| Iter | Dataset | Description | Status |
|------|---------|-------------|--------|
| 1 | **MNIST-Video** | Stack N frames of same digit class → short video. Simplest temporal dataset, no new data required. Rate and temporal variants. | ❌ |
| 2 | **NMNIST / N-MNIST** | Neuromorphic MNIST via DVS sensor. Event-based; natural fit for temporal coding. Use tonic library. | ❌ |
| 3 | **Artificial EEG** | Record SNN-FFNN layer activations across full test set → PCA/random projection → 128-channel "EEG" signal. The actual input distribution the ANP will train on. | ❌ |
| 4 | **Game of Life** | Procedurally generated Conway's Game of Life sequences. Tests whether ANP can learn non-trivial spatiotemporal rules from first principles. | ❌ |
| 5 | **DVS / event camera datasets** | Real neuromorphic video (N-Caltech101, DVS-Gesture etc.). Higher complexity temporal benchmark. | ❌ |

---

## Phase E — Spiking Predictive Coding / SPCNN (`anp_spcnn`)

The core architectural target. Combines LIF spiking neurons with PC enc-dec inference.
Every experiment runs both rate and temporal encoding for direct comparison.

### E1 — SPCNN on Static MNIST

Architecture validation before adding temporal complexity.

| Iter | Model | Encoding | Dataset | Status |
|------|-------|----------|---------|--------|
| 1 | SPC-FFNN | Rate | MNIST | ❌ |
| 2 | SPC-FFNN | TTFS | MNIST | ❌ |
| 3 | SPC-EncDec | Rate | MNIST | ❌ |
| 4 | SPC-EncDec | TTFS | MNIST | ❌ |

### E2 — SPCNN on Temporal Sequences

Next-frame prediction — the actual ANP training objective.

| Iter | Model | Encoding | Dataset | Status |
|------|-------|----------|---------|--------|
| 5 | SPC-FFNN | Rate | MNIST-Video | ❌ |
| 6 | SPC-FFNN | TTFS | MNIST-Video | ❌ |
| 7 | SPC-EncDec | Rate | MNIST-Video | ❌ |
| 8 | SPC-EncDec | TTFS | MNIST-Video | ❌ |
| 9 | SPC-EncDec | Rate | NMNIST | ❌ |
| 10 | SPC-EncDec | TTFS | NMNIST | ❌ |

---

## Phase F — ANP Assembly (`artificial_neural_prostheses`)

Connect the SPCNN to a source network via the population-activity interface.
The ANP receives 128-channel activity from the source SNN and predicts the next frame.

| Iter | Experiment | Status |
|------|-----------|--------|
| 2 | Artificial EEG construction (record source SNN → 128-ch stream) | ❌ |
| 3 | ANP v1: SPC-EncDec (rate) trained on Artificial EEG next-frame prediction | ❌ |
| 4 | ANP v2: SPC-EncDec (TTFS) trained on Artificial EEG next-frame prediction | ❌ |
| 5 | ANP on NMNIST streams — generalisation test | ❌ |
| 6 | ANP on Game of Life — non-trivial spatiotemporal rules | ❌ |

---

## Phase G — ANP Intervention (`artificial_neural_prostheses`)

Once prediction is validated, test whether the ANP can actively correct anomalous
source network activity. Introduce synthetic anomalies (lesions, noise injection,
ablations) and measure whether the ANP's corrective signal recovers normal activity.

| Iter | Experiment | Status |
|------|-----------|--------|
| 7 | Anomaly injection protocol design | ❌ |
| 8 | ANP intervention: correction of synthetic lesions | ❌ |
| 9 | ANP intervention: correction of noise-corrupted activity | ❌ |

---

## Current Status Summary (2026-04-12)

### Running
- `anp_pcnn` iter 8: PC-EncDec v2 @ 60 epochs (job 7171099)

### Needs attention
- `anp_pcnn` iter 5: PC-RNN — results exist but experiment is stuck at `experiment`
  status, unanalysed. Failed experiment (random performance). Needs conclude.

### Next actionable steps (in order)
1. Conclude `anp_pcnn` iter 5 (PC-RNN failure — document root cause)
2. Analyse `anp_pcnn` iter 8 when complete
3. Initialise `anp_snn` project — start iter 1 (SNN-CNN, rate coding)
4. Initialise `anp_datasets` project — start iter 1 (MNIST-Video)
5. Initialise `anp_spcnn` project — plan iter 1 (SPC-FFNN, rate, static MNIST)

---

## Key Open Questions

1. **Why did PC-RNN fail?** Greedy per-timestep PC inference + BPTT may create
   conflicting gradients. Fix before attempting spiking PC-RNN.
2. **At what task complexity does temporal coding beat rate coding?** The crossover
   point (static vs temporal tasks) is the central empirical finding of Phase C.
3. **Does the generative decoder in PC-EncDec close to PC-FFNN at 60 epochs?**
   (iter 8, running)
4. **Can a rate-coded SPCNN match PC-EncDec on static MNIST?** Validates the
   spiking integration before temporal tasks.
5. **What population interface compression (PCA vs random projection vs learned)
   gives the cleanest 128-ch artificial EEG?**
