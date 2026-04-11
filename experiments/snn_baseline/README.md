# SNN Baseline — Phase 1

Feedforward spiking neural network (SNN) trained on MNIST using
[snntorch](https://snntorch.readthedocs.io/). This is the **source network**
for the Artificial Neural Prostheses (ANP) project.

---

## Architecture

| Layer    | Type                  | Size |
|----------|-----------------------|------|
| Input    | Rate-coded spikes     | 784  |
| Hidden 1 | LIF (Leaky I&F)       | 512  |
| Hidden 2 | LIF (Leaky I&F)       | 256  |
| Output   | LIF (Leaky I&F)       |  10  |

* **Encoding:** Poisson rate coding — each pixel intensity becomes a Bernoulli
  spike probability repeated over T timesteps (default T = 25).
* **Training:** BPTT with snntorch fast-sigmoid surrogate gradient.
* **Loss:** Cross-entropy on the final-timestep membrane potential of the
  output layer.

---

## Quick Start

Install dependencies from the repo root first:

```bash
pip install -r requirements.txt
```

### 1 — Train

```bash
# From the repo root
python experiments/snn_baseline/train.py --epochs 20 --device cuda
```

| Flag          | Default | Description                       |
|---------------|---------|-----------------------------------|
| `--epochs`    | 10      | Number of training epochs         |
| `--timesteps` | 25      | Simulation timesteps T            |
| `--batch-size`| 128     | Mini-batch size                   |
| `--lr`        | 1e-3    | Adam learning rate                |
| `--device`    | cuda    | `cuda` or `cpu`                   |

The best checkpoint is saved to:
```
experiments/snn_baseline/checkpoints/snn_baseline_best.pth
```

### 2 — Record activations

```bash
python experiments/snn_baseline/record.py --device cuda
```

This loads the best checkpoint, runs the full MNIST test set through the
model, and saves per-sample spike trains and membrane potentials to:

```
experiments/snn_baseline/recordings/
  spk1.pt   (10000, T, 512)   hidden-layer-1 spike trains
  mem1.pt   (10000, T, 512)   hidden-layer-1 membrane potentials
  spk2.pt   (10000, T, 256)   hidden-layer-2 spike trains
  mem2.pt   (10000, T, 256)   hidden-layer-2 membrane potentials
  spk3.pt   (10000, T, 10)    output-layer   spike trains
  mem3.pt   (10000, T, 10)    output-layer   membrane potentials
  labels.pt (10000,)          ground-truth digit labels
```

These recordings are the direct input for **Phase 2** (Artificial EEG dataset).

---

## Expected Performance

With default hyperparameters (20 epochs, T=25, batch=128, lr=1e-3) you should
achieve **≥ 95% test accuracy** on MNIST.

---

## Files

| File          | Purpose                                      |
|---------------|----------------------------------------------|
| `model.py`    | SNNBaseline architecture (snntorch LIF)      |
| `train.py`    | Training script — saves best checkpoint      |
| `record.py`   | Records layer activations on the test set    |
| `checkpoints/`| Saved model weights (created by train.py)    |
| `recordings/` | Layer-wise activations (created by record.py)|
