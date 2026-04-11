# Artificial Neural Prostheses (ANP)

## Background

The brain compensates for localised damage through neural plasticity — rerouting function around
lesions, redistributing representations, and recruiting neighbouring tissue. Artificial Neural
Prostheses (ANPs) aim to replicate this externally: a small auxiliary network that monitors a
target network's activity, learns its internal dynamics, and intervenes when function degrades —
analogous to cochlear implants, retinal prosthetics, or deep-brain stimulators, but operating at
the level of learned representations rather than raw sensory transduction.

The critical constraint is **biological plausibility**: the ANP must operate under the same
constraints as biological neural circuits — local learning rules, no non-local gradient
propagation, spike-based communication, and energy-efficient inference.

---

## Final Goal: Spiking Predictive Coding Neural Network (SPCNN)

The end-state ANP is a **Spiking Predictive Coding Neural Network** with a **U-NET-like
encoder-decoder structure**:

```
Target Network Activity (all layers)
         │
         ▼
  ┌──────────────┐
  │  Population  │  ← uniform proportional sampling across all layers
  │  Interface   │    aggregate → add noise → single low-dim channel per timestep
  └──────┬───────┘    (analogous to EEG: bandwidth-limited, noisy, mixed signal)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │           SPCNN Auxiliary                │
  │                                          │
  │   Encoder (downsampling, LIF neurons)    │
  │       ↓  skip connections  ↑             │
  │   Decoder (upsampling, LIF neurons)      │
  │                                          │
  │   Learning: STDP / Hebbian / PC energy   │
  │   Objective: predict next-step activity  │
  └──────────────────────────────────────────┘
         │
         ▼
  Predicted next-step target network activity
  → used to detect anomalies / restore lost function
```

### Architecture Variants (backbone of enc-dec)
- **SPCNN-FFNN**: fully-connected LIF enc-dec
- **SPCNN-RNN**: spiking recurrent enc-dec (liquid state / STDP-RNN)
- **SPCNN-Attention**: WTA lateral inhibition or Hopfield attractor as attention analogue

### Biological Plausibility Requirements
The final SPCNN must satisfy all of the following:

| Requirement | Rationale |
|-------------|-----------|
| No backpropagation | Weight transport problem — non-local, biologically implausible |
| No weight sharing (no CNNs) | No evidence for shared synaptic weights across cortical columns |
| Local learning rules only | Synaptic updates depend only on pre/post activity (Hebbian, STDP) |
| Spike-based communication | LIF neurons, discrete spike events |
| Energy-based inference | Predictive coding: minimise free energy via local relaxation, not backprop |

> **Note:** CNNs, standard Transformers, VAEs, and BPTT-trained SNNs appear as *comparison
> baselines* only. They are not part of the bio-plausible pipeline.

---

## Roadmap

The research is structured as a series of projects, each adding one layer of complexity.
The ordering is deliberate: get working baselines with standard training first, then introduce
spiking dynamics, then remove backprop entirely.

---

### Phase A — Rate-Coded Foundations (standard backprop)

#### Project 2 — `anp_rnn`: Recurrent Neural Networks
*Establishes sequential processing baselines before introducing spiking dynamics.*

| Iter | Goal |
|------|------|
| 1 | LSTM on sequential MNIST (pixels as time steps, establish RNN data pipeline) |
| 2 | GRU baseline (compare with LSTM) |
| 3 | Vanilla RNN (Elman network, simplest recurrent unit) |

---

#### Project 3 — `anp_pcnn`: Predictive Coding Neural Networks
*Energy-based hierarchical inference — the conceptual core of ANP.*
*Uses standard gradient descent for inference (not yet bio-plausible), establishes the framework.*

| Iter | Goal |
|------|------|
| 1 | PC-FFNN: Rao & Ballard (1999) hierarchical PC — error units + representation units, inference via gradient descent on energy |
| 2 | PC-RNN: temporal predictive coding — predict next state, recurrent PC cells |
| 3 | PC enc-dec: hierarchical encoder-decoder with top-down predictions and skip connections (proto-SPCNN, rate-coded) |

---

### Phase B — Spiking Dynamics (surrogate gradients, not yet bio-plausible)

#### Project 4 — `anp_snn_rnn`: Spiking Recurrent Neural Networks
*Adds recurrent dynamics to spiking neurons. Still trained with surrogate gradients.*

| Iter | Goal |
|------|------|
| 1 | Spiking RNN — LIF neurons with recurrent connections (snntorch + BPTT) |
| 2 | Spiking LSTM analogue — adaptive threshold LIF as gating mechanism |
| 3 | Liquid State Machine — fixed random recurrent reservoir, trained readout only |

---

#### Project 5 — `anp_snn_pcnn`: Spiking Predictive Coding (Spiking + PC, surrogate grads)
*Combines spiking neurons with the PC energy framework. Surrogate gradients still used.*

| Iter | Goal |
|------|------|
| 1 | Spiking PC-FFNN — LIF error and representation units, energy minimisation |
| 2 | Spiking PC enc-dec — U-NET-shaped SPCNN, surrogate gradient training |
| 3 | Activity prediction task — train on interface recordings, predict next-step spike patterns |

---

### Phase C — Backprop-Free Learning

#### Project 6 — `anp_learning_rules`: Bio-plausible Learning Rules
*Implements the learning rules that replace backprop throughout the rest of the pipeline.*

| Iter | Goal |
|------|------|
| 1 | Hebbian learning + Oja's rule (weight normalisation, no runaway potentiation) |
| 2 | STDP — pre/post spike timing windows, LTP/LTD curves, configurable time constants |
| 3 | BCM rule — sliding threshold for selectivity without fixed normalisation |

---

#### Project 7 — `anp_snn_stdp`: SNN with STDP (replace surrogate gradients)
*Revisits the SNN baseline and RNN with purely bio-plausible learning.*

| Iter | Goal |
|------|------|
| 1 | SNN-FFNN with STDP (compare against Phase 1 surrogate gradient baseline) |
| 2 | Spiking RNN with STDP (compare against Project 4) |
| 3 | PC energy + STDP — pure local learning, no backprop anywhere |

---

### Phase D — Transformers and Bio-plausible Attention

#### Project 8 — `anp_transformer`: Vision Transformers (comparison baselines)
*Standard Transformer and ViT — not bio-plausible, used for comparison only.*

| Iter | Goal |
|------|------|
| 1 | ViT on CIFAR-10 (patch embeddings + multi-head attention, backprop) |
| 2 | Spiking ViT — LIF neurons in attention blocks (surrogate gradients) |

---

#### Project 9 — `anp_attention_bio`: Bio-plausible Attention
*WTA and Hopfield networks as biologically grounded attention analogues.*

| Iter | Goal |
|------|------|
| 1 | Winner-Take-All (WTA) lateral inhibition network — competitive attention |
| 2 | Continuous Hopfield network — attractor dynamics as content-addressable memory |
| 3 | Spiking WTA + Hopfield with STDP |

---

### Phase E — Enc-Dec / Generative (comparison baselines)

#### Project 10 — `anp_autoencoder`: Autoencoders, U-NETs, VAEs
*Standard enc-dec architectures for comparison. CNN-based, backprop.*

| Iter | Goal |
|------|------|
| 1 | FFNN autoencoder (MNIST reconstruction) |
| 2 | CNN U-NET (skip connections, encoder-decoder) |
| 3 | VAE (reparameterisation trick, latent sampling) |

---

### Phase F — Integration

#### Project 11 — `anp_interface`: Population Neural Interface
*The EEG-like bridge between target network and SPCNN.*

| Iter | Goal |
|------|------|
| 1 | Uniform proportional population sampler — sample activity at each layer proportionally |
| 2 | Aggregation + noise injection (Brown noise, configurable SNR) |
| 3 | Continuous streaming pipeline — feed live SNN layer activations as timestamped channels |

---

#### Project 12 — `anp_spcnn`: Full SPCNN Auxiliary (bio-plausible)
*Synthesises Phases A–F. The primary research contribution.*

| Iter | Goal |
|------|------|
| 1 | SPCNN-FFNN: PC enc-dec, LIF neurons, STDP, next-step activity prediction via interface |
| 2 | SPCNN-RNN: recurrent backbone (STDP-RNN or LSM) |
| 3 | SPCNN-Attention: WTA/Hopfield backbone |
| 4 | Bio-plausibility audit — verify no backprop, no weight sharing, all local rules |
| 5 | Ablation: backbone × learning rule × noise level |

---

#### Project 13 — `anp_continual_learning`: Neural Function Restoration
*End-to-end evaluation of the ANP in a continual learning setting.*

| Iter | Goal |
|------|------|
| 1 | Catastrophic forgetting baseline — target SNN on sequential tasks, measure degradation |
| 2 | ANP-assisted restoration — SPCNN monitors target, intervenes on degraded layers |
| 3 | Full evaluation — accuracy recovery curves, intervention latency, biological plausibility metrics |

---

## Dependency Graph

```
ANP-001 (SNN Baseline) ──────────────────────────────────────────────────────────────┐
                                                                                       │
Phase A:  anp_rnn (P2) ──────────────────────────────────────────────────────────┐    │
          anp_pcnn (P3) ──────────────────────────────────────────────────────┐   │    │
                                                                               │   │    │
Phase B:  anp_snn_rnn (P4) ───────────────────────────────────────────────┐   │   │    │
          anp_snn_pcnn (P5) ───────────────────────────────────────────┐   │   │   │    │
                                                                        │   │   │   │    │
Phase C:  anp_learning_rules (P6) ──► anp_snn_stdp (P7) ──────────┐    │   │   │   │    │
                                                                    │    │   │   │   │    │
Phase D:  anp_transformer (P8) ──► anp_attention_bio (P9) ──────┐   │    │   │   │   │    │
                                                                  │   │    │   │   │   │    │
Phase E:  anp_autoencoder (P10) ─────────────────────────────┐   │   │    │   │   │   │    │
                                                              │   │   │    │   │   │   │    │
Phase F:  anp_interface (P11) ────────────────────────────────────────────────────────┘    │
          anp_spcnn (P12) ◄───────────────────────────────────┘   │   │    │   │   │       │
                    ◄─────────────────────────────────────────────┘   │    │   │   │       │
                    ◄─────────────────────────────────────────────────┘    │   │   │       │
                    ◄──────────────────────────────────────────────────────┘   │   │       │
                    ◄──────────────────────────────────────────────────────────┘   │       │
                    ◄──────────────────────────────────────────────────────────────┘       │
          anp_cl (P13) ◄── anp_spcnn (P12) + target SNN (ANP-001) ◄────────────────────────┘
```

---

## Current Status

| Project | Status | Key Result |
|---------|--------|-----------|
| ANP-001 SNN Baseline | **completed** | 97.62% ± 0.12% (MNIST, 5 seeds) |
| anp_rnn (P2) | planned | — |
| anp_pcnn (P3) | planned | — |
| anp_snn_rnn (P4) | planned | — |
| anp_snn_pcnn (P5) | planned | — |
| anp_learning_rules (P6) | planned | — |
| anp_snn_stdp (P7) | planned | — |
| anp_transformer (P8) | planned | — |
| anp_attention_bio (P9) | planned | — |
| anp_autoencoder (P10) | planned | — |
| anp_interface (P11) | planned | — |
| anp_spcnn (P12) | planned | — |
| anp_cl (P13) | planned | — |
