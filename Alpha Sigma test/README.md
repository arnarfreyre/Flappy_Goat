# Alpha Sigma Test — T-alpha Activation Function

Experiments with **T-alpha**, a learnable composite activation system applied to PPO reinforcement learning for Flappy Bird.

## What is T-alpha?

T-alpha is not a single activation function you swap in for ReLU or tanh. The core idea is to build an activation from multiple **talpha-modulated basis functions** — `talpha(z)*cos(z)`, `talpha(z)*abs(z)`, `talpha(z)*z²`, etc. — and let the network learn a per-neuron weighted combination of them. Each neuron discovers its own activation shape by adjusting how much of each talpha*basis it uses.

The `AdaptiveLayer` gives every neuron a weight vector over these bases. During training the model learns not just *what* to compute, but *how to activate* — choosing its own nonlinearity from the space spanned by the talpha-modulated functions.

## The T-alpha Gate

The underlying talpha function is a parameterized sigmoid-like gate:

```
h(x) = c * (tanh(u / 2) - 1) + b

where u = a * x / 2 - d * a
```

| Parameter | Role | Learnable | Default |
|-----------|------|-----------|---------|
| `a` | Controls step sharpness (precision) | No (fixed) | varies by experiment |
| `b` | Vertical offset (shifts output up/down) | Yes | 1.0 |
| `c` | Output scale (`b*c` = distance between min and max) | Yes | 0.5 |
| `d` | Horizontal offset (x-location of the step) | Yes | 0.0 |

This gate multiplies each basis function, giving the network learnable control over when and how strongly each basis contributes.

## Evolution of the AdaptiveLayer

### Version 1 — Mixed bases (`adaptive_model.py`, `a=100,000`)

Only one basis uses talpha; the rest are plain mathematical functions. Weights initialized with `w_linear = 1.0`, all others `0.0` (starting as identity):

| Basis | Expression |
|-------|-----------|
| Linear | `z` |
| Quadratic | `z^2` |
| Cosine | `cos(z)` |
| Sine | `sin(z)` |
| T-alpha gate | `talpha(z) * z` |

### Version 2 — All-Talpha bases (`Flappy_Talpha.ipynb`, `a=20`)

Every basis is modulated through talpha, so the gating mechanism has full control over all components. Weights initialized with `w_talpha = 0.6`, all others `0.1` — the model starts with a slight bias toward the pure gate but actively uses all bases from the beginning:

| Basis | Expression |
|-------|-----------|
| T-alpha | `talpha(z)` |
| T-alpha quadratic | `talpha(z) * z^2` |
| T-alpha cosine | `talpha(z) * cos(z)` |
| T-alpha abs | `talpha(z) * abs(z)` |
| T-alpha linear | `talpha(z) * z` |

The softer `a=20` produces a smooth sigmoid-like gate (vs the near-binary step at `a=100,000`), which improves gradient flow during RL training. The key shift from v1 to v2 is that the model no longer has any "raw" basis functions — everything passes through talpha, so the network must learn to use the talpha gate to shape every component of its activation.

## Implementations

### `talpha.py` — Standalone NumPy function

Minimal reference implementation of `mfunc2()` with timing. Demonstrates the core formula in isolation.

### `custom_network.py` — NumPy network with manual backprop

A from-scratch neural network (`Network` class) that supports T-alpha as an activation alongside ReLU and tanh. Includes:
- Manual forward pass with per-layer T-alpha parameters (`c`, `d`)
- Hand-derived backpropagation computing gradients for `c` and `d`
- Gradient clipping and weight save/load
- Demo: fitting `y = x^2` with a 1 → 4 → 1 network

### `normal_model.py` — PyTorch `TAlpha` module

PyTorch `nn.Module` implementation with learnable `nn.Parameter` tensors for `c`, `d`, `b`. Compares a `talphaNetwork` (using `talpha(z) * z` gating) against a baseline `simpleNetwork` (ReLU) on the `y = x^2` regression task.

### `adaptive_model.py` — Adaptive basis function layer (v1)

Defines the original `AdaptiveLayer` with mixed bases (see Version 1 above). Each neuron has a learned weight vector over the 5 bases, initialized biased toward linear (`w_0 = 1.0`). Compares ReLU, T-alpha, and Adaptive networks on noisy `y = x^2`.

### `Flappy_Talpha.ipynb` — PPO training on Flappy Bird (v2)

Full PPO (Proximal Policy Optimization) pipeline using the all-Talpha `AdaptiveLayer` (Version 2) in the policy/value network. Each hidden layer is followed by an `AdaptiveLayer` of matching size, with actor (softmax) and critic heads on top. Layer sizes and depth vary between runs.

**Training details:**
- 8 game-state features (player position, velocity, pipe positions), normalized
- GAE (Generalized Advantage Estimation) with `gamma=0.99`, `lambda=0.95`
- Clipped surrogate objective (`epsilon=0.2`)
- Linear learning rate annealing from `3e-4` to 0
- Minibatch SGD over collected rollouts (~8192 steps per epoch, minibatch size 512)
- Gradient clipping at 0.5
- Weights saved to `Weights/AT_L2.pt`

### `Greedy_Talpha_flappy.py` — Greedy evaluation

Loads a trained `PPO_Flappy` model (v1 mixed bases, `a=20`) and runs a single greedy game — always picking the highest-probability action (argmax instead of sampling). Reports pipe count milestones every 1,000 pipes. Uses `Ad1.pt` weights.

### `weight_analysis.py` — Visualization of learned activations

Analyzes a trained all-Talpha model (`AT1.pt`) across all 3 adaptive layers. Produces two figures:
1. **Adaptive Layer Overview (3x3 grid)** — Per-layer basis weight heatmap, combined activation curves for all 64 neurons, and mean basis importance bar charts.
2. **T-alpha Gating (1x3)** — The learned `talpha(z) * z` gate shape for all 64 neurons in each layer.

### `test_network.py` — NumPy network evaluation

Loads a pretrained NumPy network from `custom_network.py` (`Weights/my_model.npz`) and plots its predictions against the true `y = x^2` curve.

### `old.py` — Early prototype

Initial minimal NumPy network prototype (no activations, incomplete backprop). Superseded by `custom_network.py`.

## Weights

| File | Model version | Description |
|------|---------------|-------------|
| `Weights/AT_L2.pt` | v2 (all-Talpha, `a=20`) | Trained with all T-alpha-modulated bases (5-layer) |
| `Weights/AT1.pt` | v2 (all-Talpha, `a=20`) | Trained with all T-alpha-modulated bases (3-layer) |
| `Weights/Ad1.pt` | v1 (mixed bases, `a=20`) | First run with original mixed bases |
| `Weights/Ad2.pt` | v1 (mixed bases, `a=20`) | Second run with original mixed bases |
| `Weights/my_model.npz` | NumPy network | Custom network trained on `y = x^2` |

## File Overview

| File | Purpose |
|------|---------|
| `talpha.py` | Reference T-alpha function (NumPy) |
| `custom_network.py` | NumPy network with manual T-alpha backprop |
| `normal_model.py` | PyTorch T-alpha module + comparison vs ReLU |
| `adaptive_model.py` | Adaptive basis layer v1 (mixed bases) |
| `Flappy_Talpha.ipynb` | PPO training for Flappy Bird (all-Talpha bases) |
| `Greedy_Talpha_flappy.py` | Greedy evaluation of a trained model |
| `weight_analysis.py` | Visualization of learned activation shapes |
| `test_network.py` | NumPy network evaluation on `y = x^2` |
| `old.py` | Early prototype (superseded) |
| `Weights/` | Saved model weights (`.pt` and `.npz`) |
