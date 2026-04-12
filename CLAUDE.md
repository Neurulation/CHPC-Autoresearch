# AutoResearch - Agent Instructions

## Kickoff

Use the `/start` command to begin or resume the autoresearch loop:

```
/start [YOUR RESEARCH TOPIC]
```

This reads state, determines what to do next, and works through each phase autonomously. For CHPC experiments, the agent tries SSH directly. If SSH isn't available, it provides manual commands.

## What This Repo Is

An automated AI research assistant that runs ML experiments on South Africa's CHPC (Center for High Performance Computing). The agent follows the scientific method in iterative research loops, tracking state so new sessions can pick up where the last left off.

## Boot Sequence (New Session)

1. Read `projects/registry.yaml` to see all projects and their status
2. Find the active project, read its `projects/{slug}/project.yaml`
3. Read the current iteration: `projects/{slug}/iterations/{NNN}/iteration.yaml`
4. Check the `status` field to determine the current phase
5. Execute that phase (see Research Phases below)
6. If the human gave specific instructions in their prompt, follow those instead

## Research Phases

Each iteration follows this sequence:

| Phase | Status Value | What To Do |
|-------|-------------|------------|
| Plan | `planned` | Read the project goal and human direction. Plan what to do. |
| Survey | `survey` | Search for relevant papers/techniques. Write notes to `literature.yaml`. |
| Implement | `implement` | Write/modify code in `src/autoresearch/`. Create experiment configs. |
| Experiment | `experiment` | Generate PBS scripts, submit to CHPC, track job IDs in `experiments.yaml`. |
| Analyze | `analyze` | Read outputs, compute statistics across seeds, write `metrics.json`. |
| Conclude | `conclude` | Write findings, update project best_result, update README leaderboard. |
| Done | `completed` | Advance to next iteration or await human direction. |

After completing a phase, update `iteration.yaml` status to the next phase.

## Running Experiments

**Locally:**
```bash
python -m autoresearch.train experiment=mnist_ffnn_adam seed=0 epochs=2 wandb.enabled=false
```

**Multi-seed sweep:**
```bash
python -m autoresearch.train --multirun experiment=mnist_ffnn_adam
```

**On CHPC:** Use `/chpc-submit` skill (agent will try SSH directly).

**Override any config at CLI:**
```bash
python -m autoresearch.train experiment=mnist_ffnn_adam seed=42 batch_size=64 epochs=20
```

## CHPC SSH (Direct Agent Access)

The agent should attempt to SSH to CHPC directly. This enables fully autonomous experiment submission.

**How it works:**
1. Read `.env` for `CHPC_USERNAME`, `CHPC_HOST`, `CHPC_LUSTRE_PATH`, `CHPC_REPO_NAME`
2. SSH to CHPC: `ssh $CHPC_USERNAME@$CHPC_HOST "command"`
3. Sync code: `ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && git pull origin develop"`
4. Submit job: `ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && qsub experiments/<name>.pbs"`
5. Check status: `ssh $CHPC_USERNAME@$CHPC_HOST "qstat -u $CHPC_USERNAME"`

**Authentication:**
- **Local Claude Code**: Uses the user's existing SSH keys (~/.ssh/)
- **Remote GitHub Copilot**: Uses SSH key stored as a GitHub secret (write to ~/.ssh/id_rsa before use)

**Fallback:** If SSH fails (no keys, network issues), provide the user with the exact commands to run manually. Never get stuck -- always have a fallback.

**SSH and `.env` in the Bash tool:** Use literal variable values from `.env` rather than `source .env`, as `source` and `export` may be blocked in sandbox environments. Read credentials with `head`/`cat` and embed them directly:
```bash
# Safe pattern — read vars then embed literals
CHPC_USER=$(grep CHPC_USERNAME .env | cut -d= -f2)
CHPC_HOST=$(grep ^CHPC_HOST .env | cut -d= -f2)
ssh ${CHPC_USER}@${CHPC_HOST} "qstat -u ${CHPC_USER}"
```

## Autonomous Monitoring (Loop)

When experiments are submitted and the agent needs to wait for CHPC results, use the `/loop` skill to set up a recurring poll rather than blocking.

**Start a monitoring loop:**
```
/loop 10m Check CHPC job status. For each completed job, extract results, write metrics.json, update state, commit+push. Resubmit any jobs that died early.
```

**Under the hood — `CronCreate`:**
- Schedules a recurring prompt at a given interval (`*/10 * * * *` for 10m)
- **Session-only by default** — the job lives only while Claude Code is open; it is lost if the session ends
- **Auto-expires after 7 days** — fires one final time then self-deletes
- Jobs only fire while the session is **idle** (never mid-query)
- The scheduler adds a small jitter (up to 10% of the period) to avoid thundering-herd

**Cancel a loop:**
```
CronDelete("<job-id>")   # job ID is returned by /loop when scheduled
```
Or just tell the agent: "stop the loop" / "cancel the monitoring".

**Standard CHPC poll loop prompt:**
```
Check CHPC job status (source .env for credentials). For each job no longer in qstat:
  1. Read per-seed outputs/train/<experiment_group>/<seed>/status.json and train.log
  2. Compute mean ± std val_acc across completed seeds
  3. Write projects/<slug>/iterations/<NNN>/metrics.json
  4. Update iteration.yaml status → completed, add analyze + conclude phases
  5. Update experiments.yaml job status → completed
  6. Update README leaderboard
  7. git add + commit + push to develop
For any job that died with incomplete seeds (fewer than 5), resubmit the PBS script
(pull develop on CHPC first so the updated walltime is picked up).
```

**PBS walltime sizing rule:** Each sequential seed adds ~N minutes of runtime. Estimate per-seed time from a smoke test or seed 0, then set `walltime = n_seeds × per_seed_time × 1.2` (20% buffer). The gpu_1 queue max is 48h. If in doubt, use 4h for RNN-class models and 2h for FFNN/CNN.

**Resume after walltime kill:** The training framework saves `last_checkpoint.pt` after every epoch. When a killed job is resubmitted, seeds that already completed will detect `start_epoch > epochs` via the checkpoint and exit immediately — no wasted compute. Seeds killed mid-run will resume from the last saved checkpoint.

## Research Roadmap: ANN → SNN Ladder

The overarching goal is a **fully biologically plausible, end-to-end spiking implementation**:
1. No backpropagation — local learning rules (Hebbian, STDP) eventually
2. Fully spike-driven — every layer emits binary spikes {0,1}; no float hidden states between layers
3. Deployable on neuromorphic / energy-efficient hardware

**Architecture ladder and snntorch primitives:**

| ANN | Fully Spiking SNN | snntorch | Status |
|-----|-------------------|----------|--------|
| FFNN | **SFNN** | `snn.Leaky` per FC layer | Done (anp_snn iter 2) ✓ |
| CNN | **SCNN** | `snn.Leaky` after each conv | Done (anp_snn iter 1) ✓ |
| Vanilla RNN | **SRNN** | `snn.RLeaky(linear_features=N)` | Planned (iter 11) |
| LSTM | **SLSTM** | `snn.SLSTM(input_size, hidden_size)` | Planned (iter 12) |
| GRU | **SGRU** | No native — custom LIF-gated GRU | Deferred |
| Transformer | **STransformer** | No native — research-level | Deferred |

**Hybrid ≠ Fully Spiking.** The anp_snn Phase B/C experiments (iters 3–10) used
a hybrid architecture: `snn.Leaky` encoder → rate-coded spike counts → standard
`nn.LSTM/GRU/RNN`. These are comparison baselines. SRNN and SLSTM need dedicated
implementations using `snn.RLeaky` and `snn.SLSTM` respectively.

**Key snntorch API:**
```python
# SRNN (snn.RLeaky — recurrent LIF, fully spiking VanillaRNN analogue)
# U[t+1] = β·U[t] + I_in[t+1] + V(S_out[t]) - R·U_thr
self.srnn = snn.RLeaky(beta=0.9, linear_features=256)  # all-to-all recurrent
spk, mem = self.srnn.init_rleaky()
for t in range(T_seq):  # T_seq=28 MNIST rows
    cur = self.fc_in(x_spk[:, t, :])  # x_spk is binary input spikes
    spk, mem = self.srnn(cur, spk, mem)  # spk is binary output

# SLSTM (snn.SLSTM — spiking LSTM cell, thresholded membrane output)
# Standard LSTM gates (σ/tanh) internally; output mem thresholded → binary spikes
self.slstm = snn.SLSTM(input_size=28, hidden_size=256)
syn, mem = self.slstm.init_slstm()
for t in range(T_seq):
    spk, syn, mem = self.slstm(x_spk[:, t, :], syn, mem)
```

**For SFNN and SCNN** (already done): `snn.Leaky` processes spikes from the previous layer
and outputs spikes to the next. The only float tensor within a layer is the membrane
potential — this is the LIF internal state, not the inter-layer signal.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/autoresearch/` | Main Python package |
| `src/autoresearch/configs/` | Hydra YAML configs |
| `src/autoresearch/configs/experiment/` | Experiment presets |
| `src/autoresearch/models/` | Model implementations |
| `src/autoresearch/utils/` | Shared utilities |
| `projects/` | Research state tracking |
| `projects/registry.yaml` | Master project index |
| `templates/` | PBS script template |
| `scripts/generate_pbs.py` | PBS script generator |
| `docs/chpc/` | CHPC documentation (offline) |
| `outputs/` | Experiment outputs (gitignored) |
| `data/` | Downloaded datasets (gitignored) |

## Adding New Components

**New model:** Create `src/autoresearch/models/mymodel.py`, add config `src/autoresearch/configs/model/mymodel.yaml` with `_target_: autoresearch.models.mymodel.MyModel`, export in `models/__init__.py`.

**New dataset:** Add config `src/autoresearch/configs/dataset/mydataset.yaml` with `_target_:` pointing to a torchvision dataset or custom class.

**New experiment:** Create `src/autoresearch/configs/experiment/name.yaml` that overrides dataset, model, optimizer, loss_fn.

## Environment Variables (.env)

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

| Variable | Purpose |
|----------|---------|
| `CHPC_USERNAME` | CHPC login username |
| `CHPC_PROJECT_ID` | PBS project code (e.g., CSCI1166) |
| `CHPC_EMAIL` | Email for PBS notifications |
| `CHPC_LUSTRE_PATH` | Lustre storage path |
| `CHPC_REPO_NAME` | Repo directory name on CHPC |
| `CHPC_MODULE_PYTHON` | Python module to load |
| `WANDB_API_KEY` | Weights & Biases API key |
| `WANDB_ENTITY` | W&B entity/team |
| `WANDB_PROJECT` | W&B project name |

## Git Workflow

- `main` -- stable releases
- `develop` -- integration branch where all code work happens

**Feature branches** (`feature/{slug}-iter-{N}-{desc}`) are used for development work (implementing new models, configs, utilities). Once the code is ready and tested locally, merge to `develop`.

**CHPC always stays on `develop`.** Experiments are distinguished by **Hydra configs**, not branches. This means multiple experiments can run concurrently on CHPC -- they use different output directories based on `experiment_group` and `seed`. Never checkout a feature branch on CHPC.

**Workflow:**
1. Agent creates a feature branch for implementation work
2. Agent writes code, configs, PBS scripts on the feature branch
3. Agent merges to `develop` when ready
4. Human (or CI) pulls `develop` on CHPC and submits PBS jobs
5. Multiple PBS jobs can run in parallel -- each writes to its own `outputs/` subdirectory

## Rules

- Never commit `.env` or any secrets
- Always update state files after completing a phase
- Test locally with `epochs=2 wandb.enabled=false` before submitting to CHPC
- Use 5 seeds for statistical robustness
- Keep experiment configs composable (dataset + model + optimizer + loss_fn)
- Outputs are gitignored; only state/configs/code are tracked
