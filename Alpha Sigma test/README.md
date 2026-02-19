# Alpha Sigma Test — T-alpha Activation Function

Experiments with **T-alpha**, a custom learnable activation function, applied to PPO reinforcement learning for Flappy Bird.

## What is T-alpha?

T-alpha is a parameterized activation function where each neuron learns its own activation shape during training. Unlike fixed activations (ReLU, tanh), T-alpha has per-neuron learnable parameters that let the network adapt its nonlinearity to the task.

## Mathematical Definition

```
h(x) = c * (tanh(u / 2) - 1) + b

where u = a * x / 2 - d * a
```

| Parameter | Role | Learnable | Default |
|-----------|------|-----------|---------|
| `a` | Controls step sharpness (precision) | No (fixed) | 100,000 |
| `b` | Vertical offset (shifts output up/down) | Yes | 1.0 |
| `c` | Output scale (`b*c` = distance between min and max) | Yes | 0.5 |
| `d` | Horizontal offset (x-location of the step) | Yes | 0.0 |

With default parameters, T-alpha produces a sharp step-like function. The learnable parameters (`c`, `d`, `b`) allow each neuron to position, scale, and shift this step independently.

## How It's Used

T-alpha is typically applied as a **gating mechanism** rather than a direct activation:

```python
z = talpha(z) * z  # element-wise gate: learned step function modulates the input
```

This lets the network learn per-neuron input gates — selectively passing or suppressing signal based on learned thresholds.

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

### `adaptive_model.py` — Adaptive basis function layer

Extends T-alpha into `AdaptiveLayer`, where each neuron learns a weighted combination of 5 basis functions:

| Basis | Expression |
|-------|-----------|
| Linear | `z` |
| Quadratic | `z^2` |
| Cosine | `cos(z)` |
| Sine | `sin(z)` |
| T-alpha gate | `talpha(z) * z` |

Each neuron has a learned weight vector over these bases, initialized biased toward linear (`w_0 = 1.0`). Compares ReLU, T-alpha, and Adaptive networks on noisy `y = x^2`.

### `flappy_ppo.ipynb` — PPO training on Flappy Bird

Full PPO (Proximal Policy Optimization) pipeline using `AdaptiveLayer` in the policy/value network:

```
PPO_Flappy architecture:
  Linear(8, 64) → AdaptiveLayer(64)
  Linear(64, 64) → AdaptiveLayer(64)
  Linear(64, 64) → AdaptiveLayer(64)
  ├─ Actor head → Linear(64, 2) → softmax (flap / no-flap)
  └─ Critic head → Linear(64, 1) (state value)
```

**Training details:**
- 8 game-state features (player position, velocity, pipe positions), normalized
- GAE (Generalized Advantage Estimation) with `gamma=0.99`, `lambda=0.95`
- Clipped surrogate objective (`epsilon=0.2`)
- Linear learning rate annealing from `5e-4` to 0
- Minibatch SGD over collected rollouts (~8192 steps per epoch)
- Weights saved to `../flappy weights/TalphaModel.pt`

## File Overview

| File | Purpose |
|------|---------|
| `talpha.py` | Reference T-alpha function (NumPy) |
| `custom_network.py` | NumPy network with manual T-alpha backprop |
| `normal_model.py` | PyTorch T-alpha module + comparison vs ReLU |
| `adaptive_model.py` | Adaptive basis layer (T-alpha as one of 5 bases) |
| `flappy_ppo.ipynb` | PPO training for Flappy Bird using AdaptiveLayer |
| `Weights/` | Saved model weights |
