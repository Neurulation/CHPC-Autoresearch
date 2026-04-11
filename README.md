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

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Status |
|---------|------|-------|-----------|-----|-------|---------|--------|
| Image Processing NN | 1 | CNN (2 conv + 1 FC) | Adam | ✗ | 5 | **99.17% ± 0.10%** | ✅ |
| Image Processing NN | 1 | FFNN (784-256-128-10) | Adam | ✗ | 5 | 98.06% ± 0.15% | ✅ |
| Artificial Neural Prostheses | 1 | SNN (784-512-256-10 LIF, T=25) | Adam | ✗ | 5 | 97.62% ± 0.12% | ✅ |

### CIFAR-10

| Project | Iter | Model | Optimizer | Aug | Seeds | Val Acc | Status |
|---------|------|-------|-----------|-----|-------|---------|--------|
| Image Processing NN | 3 | ResNet-18 | SGD+Cosine | ✓ | 5 | **94.96% ± 0.38%** | ✅ |
| Image Processing NN | 3 | ResNet-18 | Adam | ✓ | 5 | 90.57% ± 0.51% | ✅ |
| Image Processing NN | 1 | ResNet-18 | Adam | ✗ | 5 | 83.56% ± 0.36% | ✅ |
| Image Processing NN | 2 | ResNet-18 | SGD+Cosine | ✗ | 5 | 78.87% ± 0.94% | ✅ |

*Updated 2026-04-11.*

**Key findings:**
- *MNIST: CNN outperforms FFNN at 99.17% vs 98.06%. SNN baseline 97.62% ± 0.12% (5 seeds) — trails dense nets as expected given rate coding overhead; all seeds early-stopped in 9-15 epochs*
- *CIFAR-10: Data augmentation was THE limiting factor. SGD+cosine with aug: 94.96% (+16.09%). Adam with aug: 90.57% (+7.01%). SGD+cosine beats Adam when both use augmentation*
- *[Iter 1 W&B](https://wandb.ai/arneschreuder/chpc_autoresearch/runs/jy2d7rv2)*

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
