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

Val Acc = mean ± std across 5 seeds. All runs: Adam optimizer, no augmentation, MNIST dataset.

### MNIST — Completed

Sorted by descending accuracy. Only valid, fully-converged results.

| # | Project | Iter | Model | Val Acc | Notes |
|---|---------|------|-------|---------|-------|
| 1 | Image Processing NN | 1 | CNN (2 conv + 1 FC) | **99.17% ± 0.10%** | spatial |
| 2 | ANP — PC-NN | 12 | PC-CNN v2 (energy schedule + clip=2.0) | **99.11% ± 0.08%** | **new best PC result; +5.99pp vs PC-CNN iter11** |
| 3 | ANP — RNN | 2 | GRU (2-layer, h=256) | **99.06% ± 0.14%** | sequential T=28 |
| 4 | ANP — RNN | 1 | LSTM (2-layer, h=256) | 98.95% ± 0.11% | sequential T=28 |
| 5 | ANP — SNN | 1 | SNN-CNN (rate, T=25) | 98.87% ± 0.13% ᴬ | fully spiking, spatial |
| 6 | ANP — SNN | 4 | Hybrid-GRU (rate, T=25) | 98.75% ± 0.15% ᴮ | hybrid baseline |
| 7 | ANP — SNN | 3 | Hybrid-LSTM (rate, T=25) | 98.68% ± 0.04% ᴮ | hybrid baseline |
| 8 | ANP — SNN | 8b | Hybrid-LSTM TTFS (threshold=0.9) ᶜ | 98.67% ± 0.20% | corrected TTFS run; ~parity with rate |
| 9 | ANP — SNN | 9b | Hybrid-GRU TTFS (threshold=0.9) ᶜ | 98.63% ± 0.19% | corrected TTFS run; -0.12pp vs rate |
| 10 | ANP — SNN | 7 | SCNN TTFS (T=25) | 98.41% ± 0.17% ᶜ | fully spiking, temporal |
| 11 | Image Processing NN | 1 | FFNN (784-256-128-10) | 98.06% ± 0.15% | dense |
| 12 | ANP — RNN | 3 | Vanilla RNN (2-layer, h=256) | 97.89% ± 0.35% | sequential T=28 |
| 13 | ANP — SNN | 2 | SNN-FFNN (rate, T=25) | 97.62% ± 0.12% ᴬ | fully spiking, dense |
| 14 | ANP — PC-NN | 3 | PC-FFNN v3 + CE + grad clip | 97.30% ± 0.22% | previous best PC result |
| 15 | ANP — PC-NN | 2 | PC-FFNN v2 + CE head | 97.21% ± 0.35% | predictive coding |
| 16 | ANP — SNN | 5 | Hybrid-VanillaRNN (rate, T=25) | 97.17% ± 0.52% ᴮ | hybrid baseline |
| 17 | ANP — PC-NN | 8 | PC-EncDec v2 @ 60ep | 97.14% ± 0.21% | best PC-EncDec |
| 18 | ANP — SNN | 6 | SFNN TTFS (T=25) | 97.12% ± 0.20% ᶜ | fully spiking, temporal |
| 19 | ANP — SNN | 10b | Hybrid-VanillaRNN TTFS (threshold=0.9) ᶜ | 96.87% ± 0.35% | corrected TTFS run; -0.31pp vs rate |
| 20 | ANP — PC-NN | 10 | PC-EncDec v2 + cosine LR | 96.75% ± 0.19% | cosine LR degraded -0.39pp vs flat |
| 21 | ANP — PC-NN | 7 | PC-EncDec v2 @ 30ep | 96.59% ± 0.28% | |
| 22 | ANP — PC-NN | 11 | PC-CNN | 93.12% ± 0.56% | first PC-CNN; underperforms PC baselines |

### MNIST — In Progress

| Project | Iter | Model | Job | Status |
|---------|------|-------|-----|--------|
| None | - | - | - | No active MNIST jobs |

### MNIST — Planned

| Project | Iter | Model | Notes |
|---------|------|-------|-------|
| ANP — SNN | 11 | **SRNN** rate — `snn.RLeaky(linear_features=256)`, T=28 ᴰ | First truly fully spiking RNN |
| ANP — SNN | 12 | **SLSTM** rate — `snn.SLSTM(28, 256)`, T=28 ᴰ | First truly fully spiking LSTM |
| ANP — SNN | 13 | SRNN TTFS — `snn.RLeaky`, T=28 ᴰ | After iter 11 |
| ANP — SNN | 14 | SLSTM TTFS — `snn.SLSTM`, T=28 ᴰ | After iter 12 |
| ANP — Contrastive | 1 | SimCLR FFNN (784→512→256) ᴱ | NT-Xent + linear probe; ANN baseline |
| ANP — Contrastive | 2 | SimCLR CNN (Conv→FC256) ᴱ | NT-Xent + linear probe; ANN baseline |
| ANP — Contrastive | 3 | SNN SimCLR FFNN (T=25 rate) ᴱ | First spiking contrastive model |
| ANP — Contrastive | 4 | SNN SimCLR CNN (T=25 rate) ᴱ | SNN vs ANN contrastive comparison |
| ANP — Contrastive | 5 | PC Contrastive (T_pc=20 free inf.) ᴱ | Addresses SPC-FFNN dead end |

### Invalid / Abandoned

Excluded from the leaderboard. Listed for traceability.

| Project | Iter | Model | Result | Reason |
|---------|------|-------|--------|--------|
| ANP — PC-NN | 9 | PC-EncDec v2 + cosine LR | ~~95.82% ± 0.20%~~ | Wrong PBS entry point + early stopping bug (ep11); fixed in iter10 |
| ANP — PC-NN | 1 | PC-FFNN v1 | ~96% / 89.0% ± 4.7% | Training-evaluation objective mismatch; superseded by iter2 |
| ANP — PC-NN | 4 | PC-FFNN v4 eps=0.01 | 93.95% ± 0.27% | Under-trained (30ep; needs ~75ep); not a real ceiling |
| ANP — PC-NN | 6 | PC-EncDec v1 | 93.13% ± 0.35% | Two architectural bugs; both fixed in iter7 |
| ANP — SNN | 8/9/10 | Hybrid-LSTM/GRU/VanillaRNN TTFS | ~11% | threshold=1.0 → LIF silent; corrected and superseded by valid iter8b/9b/10b results |
| ANP — SPCNN | 2–5 | SPC-FFNN v1/v2/A/D | ~11% | Shared SNN/PC weights — architecturally incompatible; all variants collapse to chance |

---

ᴬ **SNN rate coding (anp_snn iters 1–2):** SNN-CNN 98.87% (−0.30pp vs non-spiking CNN 99.17%); SNN-FFNN 97.62% (−0.44pp vs non-spiking FFNN). Rate coding with LIF neurons is near-lossless for MNIST — encoding overhead is minimal, not information loss.

ᴮ **Phase B hybrid baselines (anp_snn iters 3–5) — all complete:** `snn.Leaky` encoder (T=25 Bernoulli per row) → spike counts → standard `nn.LSTM/GRU/RNN`. **Hybrid, not fully spiking.** Results: GRU 98.75% (−0.31pp vs non-spiking), LSTM 98.68% (−0.27pp), VanillaRNN 97.17% (−0.72pp). Gated architectures absorb spike encoding loss (−0.27–0.31pp gap); plain RNN is 2.3× more sensitive (−0.72pp gap, amplified by vanishing gradients + sparse spikes).

ᶜ **Phase C TTFS (anp_snn iters 6–10):** TTFS is consistently WORSE than rate across all tested architectures: SFNN −0.49pp (iter6), SCNN −0.46pp (iter7), Hybrid-LSTM −0.01pp (iter8b), Hybrid-GRU −0.12pp (iter9b), Hybrid-VanillaRNN −0.31pp (iter10b). LIF summation largely discards spike timing order, so TTFS behaves like a binarized proxy of input intensity and offers no gain over rate coding here.

ᴰ **Fully spiking recurrent — planned (anp_snn iters 11–14):** SRNN uses `snn.RLeaky(linear_features=256)` over T=28 rows — binary spikes throughout, no standard RNN cells. SLSTM uses `snn.SLSTM(28, 256)` — standard LSTM gates internally, but thresholded membrane → binary output spikes. First genuinely fully spiking recurrent models in the ladder.

ᴱ **Contrastive learning (anp_contrastive iters 1–5):** SimCLR-style NT-Xent pre-training followed by linear probe evaluation. Metric reported is linear probe val_accuracy (frozen encoder + trained linear head). Iters 1–2 (ANN): MLP and CNN baselines. Iters 3–4 (SNN): first fully spike-driven self-supervised representation learners. Iter 5 (PC+Contrastive): PC-FFNN encoder trained with NT-Xent — addresses the SPC-FFNN shared-weight dead end. Supervised comparison baselines: FFNN 97.30%, CNN 99.17%, SNN-FFNN 97.62%, SNN-CNN 98.87%.

⁸ **PC-EncDec v2 @ 60ep (iter8):** 97.14% ± 0.21% (+0.55pp vs 30ep). Training budget alone closed 78% of the gap to PC-FFNN v3. Seeds 0/3/4 needed all 60 epochs; slow convergence is the main bottleneck. Flat LR remains optimal — cosine LR (iter10) degraded performance by −0.39pp.

¹⁰ **PC-EncDec v2 + cosine LR (iter10):** 96.75% ± 0.19% — cosine LR (T_max=60, eta_min=1e-6) degraded −0.39pp vs flat LR iter8 (97.14%). PC models are LR-sensitive: aggressive decay stalls slow inference-phase credit assignment. Flat LR (iter8) remains best PC-EncDec; still −0.55pp behind PC-FFNN v3.



### CIFAR-10

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Status |
|---------|------|-------|-----------|-----|-------|---------|--------|
| Image Processing NN | 3 | ResNet-18 | SGD+Cosine | ✓ | 5 | **94.96% ± 0.38%** | ✅ |
| Image Processing NN | 3 | ResNet-18 | Adam | ✓ | 5 | 90.57% ± 0.51% | ✅ |
| Image Processing NN | 1 | ResNet-18 | Adam | ✗ | 5 | 83.56% ± 0.36% | ✅ |
| Image Processing NN | 2 | ResNet-18 | SGD+Cosine | ✗ | 5 | 78.87% ± 0.94% | ✅ |

*Updated 2026-04-13.*

**Key findings:**
- *MNIST: CNN outperforms FFNN at 99.17% vs 98.06%. SNN-CNN (iter1, anp_snn) 98.87% ± 0.13% — only -0.30pp behind non-spiking CNN, confirming LIF neurons + rate coding are effective for spatial feature extraction. SNN-FFNN baseline (iter2, anp_snn) 97.62% ± 0.12% — -0.44pp vs non-spiking FFNN despite larger hidden dims (512-256 vs 256-128). Architecture gain: +1.25pp from adding convolutional spiking layers.*
- *SNN Phase B (iters 3-5, anp_snn) — hybrid spiking recurrent results (all complete): SNN-GRU 98.75% ± 0.15% (iter4), SNN-LSTM 98.68% ± 0.04% (iter3), SNN-VanillaRNN 97.17% ± 0.52% (iter5). Rate-coded spiking input is near-lossless for gated recurrent models: −0.27pp (LSTM) and −0.31pp (GRU) gap vs non-spiking counterparts. VanillaRNN gap −0.72pp is 2.3× larger — demonstrates that gating (LSTM/GRU gates filter + compress input signal) absorbs spike encoding loss; plain RNN without gating is more sensitive to input precision. High variance for VanillaRNN (0.52pp std vs 0.04-0.15pp for LSTM/GRU) amplified by interaction of vanishing gradients with sparse spike inputs.*
- *SNN Phase C TTFS (iters 6-7, anp_snn) — TTFS is consistently WORSE than rate coding for static MNIST: SFNN TTFS 97.12% ± 0.20% vs rate 97.62% (−0.49pp, iter6); SCNN TTFS 98.41% ± 0.17% vs rate 98.87% (−0.46pp, iter7). Remarkably consistent ~0.47pp penalty across two different architectures points to the encoding itself, not the downstream model, as the cause. Mechanism: LIF spike-count integration discards temporal order information — TTFS spike-count = binarized image (pixel fires or not within T steps), whereas rate coding preserves graded intensity via Bernoulli sampling. For static MNIST, graded intensity > spike timing.*
- *GRU (iter 2) beats LSTM (iter 1): 99.06% ± 0.14% vs 98.95% ± 0.11%, with 25% fewer parameters (617k vs ~821k). Gate reduction (4→3 gates) did not hurt — confirms GRU parity with LSTM on seq-MNIST (Chung et al. 2014).*
- *Vanilla RNN (iter 3): 97.89% ± 0.35% — far better than predicted. Literature expects 10-20pp regression from LSTM for T>>10 (Bengio et al. 1994); actual gap from GRU is only 1.17pp. Adam's adaptive LR compensates for vanishing gradients at T=28, acting as a significant equaliser. Completes the RNN trilogy: GRU (99.06%) → LSTM (98.95%) → Vanilla (97.89%). Parameter efficiency: 207k vs 617k (GRU) for 1.17pp.*
- *LSTM/GRU on sequential MNIST (T=28): competitive with CNN despite processing pixels row-by-row.*
- *PC-CNN v2 (iter 12): 99.11% ± 0.08% — major stabilization breakthrough over PC-CNN iter11 (93.12% ± 0.56%, +5.99pp). Normalized energy scheduling (max=0.1, warmup=10) plus relaxed gradient clipping (2.0) removed optimization suppression and made PC-CNN the strongest predictive-coding model in this repository, narrowly trailing ANN-CNN by only 0.06pp.*
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
