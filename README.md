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

## Leaderboard

Results grouped by dataset. Val Acc = mean ± std across seeds where available. ⚠️ = preliminary (single seed / partial run).

### MNIST

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Notes | Status |
|---------|------|-------|-----------|-----|-------|---------|-------|--------|
| Image Processing NN | 1 | CNN (2 conv + 1 FC) | Adam | ✗ | 5 | **99.17% ± 0.10%** | spatial | ✅ |
| ANP — RNN | 2 | GRU (2-layer, h=256) | Adam | ✗ | 5 | **99.06% ± 0.14%** | sequential (T=28) | ✅ |
| ANP — RNN | 1 | LSTM (2-layer, h=256) | Adam | ✗ | 5 | 98.95% ± 0.11% | sequential (T=28) | ✅ |
| Image Processing NN | 1 | FFNN (784-256-128-10) | Adam | ✗ | 5 | 98.06% ± 0.15% | dense | ✅ |
| ANP — RNN | 3 | Vanilla RNN (2-layer, h=256) | Adam | ✗ | 5 | 97.89% ± 0.35% | sequential (T=28) | ✅ |
| Artificial Neural Prostheses | 1 | SNN (784-512-256-10 LIF, T=25) | Adam | ✗ | 5 | 97.62% ± 0.12% | rate-coded | ✅ |
| ANP — PC-NN | 3 | PC-FFNN v3 + CE head + grad clip (784-256-128-10) | Adam | ✗ | 5 | **97.30% ± 0.22%**⁴ | predictive coding | ✅ |
| ANP — PC-NN | 2 | PC-FFNN v2 + CE head (784-256-128-10) | Adam | ✗ | 5 | **97.21% ± 0.35%**³ | predictive coding | ✅ |
| ANP — PC-NN | 8 | PC-EncDec v2 @ 60ep (784-256-128 enc+dec, β=0.1) | Adam | ✗ | 5 | **97.14% ± 0.21%**⁸ | generative PC, 60ep ceiling | ✅ |
| ANP — PC-NN | 7 | PC-EncDec v2 (784-256-128 enc+dec, β=0.1) | Adam | ✗ | 5 | **96.59% ± 0.28%**⁷ | generative PC, ff-forward eval | ✅ |
| ANP — PC-NN | 1 | PC-FFNN (784-256-128-10) | Adam | ✗ | 5 | ~96%¹ / 89.0% ± 4.7%² | predictive coding | ⚠️ |
| ANP — PC-NN | 4 | PC-FFNN v4 + eps=0.01 (784-256-128-10) | Adam (eps=0.01) | ✗ | 5 | 93.95% ± 0.27%⁵ | not converged | ⚠️ |
| ANP — PC-NN | 6 | PC-EncDec v1 (784-256-128 enc+dec) | Adam | ✗ | 5 | 93.13% ± 0.35%⁶ | train/val mismatch + β=1 | ⚠️ |

¹ Estimated best-epoch val_acc (~epoch 3) based on smoke test; true best-epoch val_acc not directly recorded.  
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

### CIFAR-10

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Status |
|---------|------|-------|-----------|-----|-------|---------|--------|
| Image Processing NN | 3 | ResNet-18 | SGD+Cosine | ✓ | 5 | **94.96% ± 0.38%** | ✅ |
| Image Processing NN | 3 | ResNet-18 | Adam | ✓ | 5 | 90.57% ± 0.51% | ✅ |
| Image Processing NN | 1 | ResNet-18 | Adam | ✗ | 5 | 83.56% ± 0.36% | ✅ |
| Image Processing NN | 2 | ResNet-18 | SGD+Cosine | ✗ | 5 | 78.87% ± 0.94% | ✅ |

*Updated 2026-04-12.*

**Key findings:**
- *MNIST: CNN outperforms FFNN at 99.17% vs 98.06%. SNN baseline 97.62% ± 0.12% — trails dense nets as expected given rate coding overhead.*
- *GRU (iter 2) beats LSTM (iter 1): 99.06% ± 0.14% vs 98.95% ± 0.11%, with 25% fewer parameters (617k vs ~821k). Gate reduction (4→3 gates) did not hurt — confirms GRU parity with LSTM on seq-MNIST (Chung et al. 2014).*
- *Vanilla RNN (iter 3): 97.89% ± 0.35% — far better than predicted. Literature expects 10-20pp regression from LSTM for T>>10 (Bengio et al. 1994); actual gap from GRU is only 1.17pp. Adam's adaptive LR compensates for vanishing gradients at T=28, acting as a significant equaliser. Completes the RNN trilogy: GRU (99.06%) → LSTM (98.95%) → Vanilla (97.89%). Parameter efficiency: 207k vs 617k (GRU) for 1.17pp.*
- *LSTM/GRU on sequential MNIST (T=28): competitive with CNN despite processing pixels row-by-row.*
- *PC-FFNN v4 (iter 4): eps=0.01 fix CONFIRMED zero energy explosions across all 5 seeds (energy monotonically decreases to ~0.21 at ep30). However 93.95% is a convergence artifact — not a performance comparison. Adam eps=0.01 reduces effective step size in late training, needing ~50-75 epochs to match the single-seed diagnostic of 98.12%. A future re-run with epochs=75 will establish the PC-FFNN ceiling.*
- *PC-FFNN v3 (iter 3): Gradient clipping (max_grad_norm=0.5) is a partial improvement — variance reduced (0.35→0.22pp), explosions delayed, mean accuracy +0.09pp to 97.30% ± 0.22%. Root cause: clipping bounds gradient magnitude but not the energy value itself.*
- *PC-FFNN v1 (iter 1): train accuracy 100% from epoch 2 via supervised clamping, but val CE stuck at ~1.54 (uncalibrated). Root cause: training-evaluation objective mismatch between clamped and free inference.*
- *PC-EncDec v2 @ 60ep (iter 8): 97.14% ± 0.21% — +0.55pp over 30ep (96.59%). Training budget alone closed 78% of the gap to PC-FFNN v3 (-0.71pp→-0.16pp). Seeds 1/2 early-stopped; seeds 0/3/4 needed all 60 epochs — slow convergence is the main bottleneck. Cosine LR decay recommended for iter 9 to accelerate convergence within 30-40 epochs.*
- *PC-EncDec v2 (iter 7): 96.59% ± 0.28% — +3.46pp vs iter 6 (93.13%). Both fixes confirmed: (1) closing the train/val distribution mismatch (pure-feedforward eval) was the dominant contributor; (2) reducing β from 1.0 to 0.1 (Y_max 0.5→0.1) shifted gradient budget to 90% CE / 10% energy. Generative decoder is now a mild regulariser, not a hindrance. All seeds best at epochs 26-30 — model not yet converged at epoch 30; iter 8 recommended at 60 epochs to establish ceiling.*
- *PC-EncDec (iter 6): 93.13% ± 0.35% ceiling caused by two compounding bugs: (1) train/val mismatch — cls_head trained on feedforward r_{L-1} but validated on inference-modified r_{L-1} (20 PC steps shift the representation distribution); (2) Y_max=0.5 = β=1 VAE — reconstruction and classification compete with equal gradient budget, known suboptimal for discrimination (Higgins et al. 2017). Iter 7 fixes both: pure-feedforward forward() + Y_max=0.1 (β=0.1, 90% CE gradient).*
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
| `/start` | **Kickoff / resume the full autoresearch loop** |
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
