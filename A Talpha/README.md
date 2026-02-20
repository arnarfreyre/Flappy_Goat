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
| `a` | Controls step sharpness (precision) — forward pass only | No (fixed) | 200 |
| `a_grad` | Controls gradient smoothness — backward pass only | No (fixed) | 2 |
| `b` | Vertical offset (shifts output up/down) | Yes | 1.0 |
| `c` | Output scale (`b*c` = distance between min and max) | Yes | 0.5 |
| `d` | Horizontal offset (x-location of the step) | Yes | 0.0 |

This gate multiplies each basis function, giving the network learnable control over when and how strongly each basis contributes.

### Straight-Through Estimator (STE)

The current implementation (`Talpha.py`) uses a **straight-through estimator** to decouple the forward and backward passes:

```python
hard = c * (tanh(a * z / 4 - d * a / 2) - 1) + b        # sharp step (large a)
soft = c * (tanh(a_grad * z / 4 - d * a_grad / 2) - 1) + b  # smooth curve (small a_grad)

return soft + (hard - soft).detach()
```

- **Forward pass** uses `a` (e.g. 200–300) producing a near-binary step — the network sees crisp on/off gating.
- **Backward pass** uses `a_grad` (e.g. 1–2) providing smooth gradients — the optimizer gets useful gradient signal everywhere.

The `.detach()` trick means `hard - soft` contributes no gradient, so gradients flow only through `soft`.

## Configuring Adaptive Layers

Each `AdaptiveLayer` accepts per-layer configuration through three groups of parameters:

### Basis weights (`w_init`)

The 5-element list controls the initial mixing weight for each talpha-modulated basis function:

```
w_init: [Ta(z), Ta(z)*z², Ta(z)*cos(z), Ta(z)*abs(z), Ta(z)*z]
```

| Index | Basis | What it contributes |
|-------|-------|---------------------|
| 0 | `Ta(z)` | Pure gate — binary on/off signal |
| 1 | `Ta(z)*z²` | Gated quadratic — amplifies large inputs |
| 2 | `Ta(z)*cos(z)` | Gated oscillation — periodic features |
| 3 | `Ta(z)*abs(z)` | Gated absolute value — symmetric magnitude |
| 4 | `Ta(z)*z` | Gated linear — dominant default, like a learnable ReLU |

Examples of initialization strategies:

```python
# Uniform — let the network decide everything from a balanced start
w_init = [0.2, 0.2, 0.2, 0.2, 0.2]

# Linear-biased — start close to a gated ReLU, learn deviations
w_init = [0.1, 0.1, 0.1, 0.1, 0.6]

# Gate-heavy — emphasize the pure on/off signal
w_init = [0.6, 0.1, 0.1, 0.1, 0.1]
```

All weights are learnable `nn.Parameter` tensors — the initial values just set the starting point. The network adjusts them during training.

### Sharpness parameters (`a` and `a_grad`)

These control the talpha gate's precision and gradient smoothness. They are **fixed per layer** (not learnable), set at initialization:

```python
a_hard1 = 200     # sharp forward step for layers 1, 4, 5
a_soft1 = 2       # smooth backward gradient for layers 1, 4, 5

a_hard23 = [200, 300]  # layers 2 and 3 can differ
a_soft23 = [1, 2]      # layers 2 and 3 gradient smoothness
```

| Parameter | Low values (1–10) | High values (100–300) |
|-----------|-------------------|-----------------------|
| `a` | Smooth sigmoid gate in forward pass | Near-binary step in forward pass |
| `a_grad` | Very smooth gradients, slow convergence | Sharper gradients, may vanish at extremes |

The ratio between `a` and `a_grad` matters — a large gap (e.g. `a=200, a_grad=2`) means the network "sees" a crisp step but "learns" through a smooth landscape.

### Per-layer config (`layer_configs`)

The `layer_configs` list lets you set different parameters for each adaptive layer:

```python
adap_l1 = [0.2, 0.2, 0.2, 0.2, 0.2]  # uniform basis weights
adap_l2 = [0.2, 0.2, 0.2, 0.2, 0.2]
adap_l3 = [0.2, 0.2, 0.2, 0.2, 0.2]
adap_l4 = [0.2, 0.2, 0.2, 0.2, 0.2]
adap_l5 = [0.2, 0.2, 0.2, 0.2, 0.2]

a_hard1 = 200
a_soft1 = 2
a_hard23 = [200, 300]
a_soft23 = [1, 2]

layer_configs = [
    {'a': a_hard1,     'a_grad': a_soft1,     'w_init': adap_l1},  # layer 1
    {'a': a_hard23[0], 'a_grad': a_soft23[0], 'w_init': adap_l2},  # layer 2
    {'a': a_hard23[1], 'a_grad': a_soft23[1], 'w_init': adap_l3},  # layer 3
    {'a': a_hard1,     'a_grad': a_soft1,     'w_init': adap_l4},  # layer 4
    {'a': a_hard1,     'a_grad': a_soft1,     'w_init': adap_l5},  # layer 5
]

model = PPO_Flappy([64, 64, 64, 64, 64], layer_configs=layer_configs)
```

When `layer_configs` is `None`, all layers share the same defaults (`a=200, a_grad=2, w_init=[0.6, 0.1, 0.1, 0.1, 0.1]`). Using per-layer configs lets you experiment with different gate sharpness or initialization at different depths — e.g. sharper gates in middle layers, smoother at the boundaries.

## Training Approaches

### Notebook-based (`Flappy_Talpha.ipynb`, `Talpha_no_batch.ipynb`)

Self-contained training notebooks with inline PPO loop. Two variants:

- **`Flappy_Talpha.ipynb`** — Batched training. Collects multiple episodes per epoch until `target_steps` is reached (default 8192), then runs minibatch SGD with gradient clipping. Uses `compute_advantage()` per episode with GAE, then concatenates for training. Architecture: `[1024, 128, 16, 256, 256]`.

- **`Talpha_no_batch.ipynb`** — Single-episode training. Runs one game per epoch, trains on that episode immediately. Simpler loop, useful for quick iteration and debugging. Supports `layer_configs` for per-layer experiments. Architecture: `[64, 64, 64]` (configurable).

### Module-based (`train_Talpha.ipynb` + `flappy_agent_Talpha.py`)

Separates the agent logic into a reusable module:

- **`flappy_agent_Talpha.py`** — `FlappyAgent` class wrapping `FlappyNetwork` (actor-critic with AdaptiveLayers). Handles episode collection, GAE computation, batched PPO training with minibatch SGD, multiple value loss functions (`mse`, `rmse`, `normalized_rmse`, `relative_rmse`), EMA value normalization, and greedy evaluation with adaptive test frequency.

- **`train_Talpha.ipynb`** — Calls `agent.run_training(...)` with a parameter dict. Saves training stats to CSV for later analysis.

```python
training_params = {
    'gamma': 0.99, 'lam': 0.95, 'clip_eps': 0.2,
    'clip_coef': 1.0, 'value_coef': 0.5, 'entropy_coef': 0.01,
    'max_grad_norm': 0.5, 'learning_rate': 0.0003,
    'ppo_epochs': 5, 'num_epochs': 1000,
    'target_steps': 8192, 'minibatch_size': 512,
    'reward_values': {'positive': 1.0, 'tick': 0.0, 'loss': -5.0},
}
agent.run_training(**training_params, test_exploit=True,
                   result_path='Results/TA_1024x128x16x256x256.csv')
```

## Implementations

### `Talpha.py` — AdaptiveLayer module (current)

The shared PyTorch module imported by all training scripts. Defines the `AdaptiveLayer` with the STE trick (`a`/`a_grad`), 5 talpha-modulated bases, and per-neuron learnable mixing weights. This is the single source of truth for the activation function.

### `flappy_agent_Talpha.py` — Modular PPO agent

Reusable agent class (`FlappyAgent`) with `FlappyNetwork` (actor-critic). Features:
- Batched episode collection with configurable `target_steps`
- GAE advantage computation (Eq 11 in PPO paper)
- Minibatch PPO training with configurable epochs and batch size
- Multiple value loss functions: `mse`, `rmse`, `normalized_rmse`, `relative_rmse`
- EMA-based value normalization
- Greedy evaluation mode (`Exploit`) with adaptive test frequency
- CSV result logging

### `Greedy_Talpha_flappy.py` — Greedy evaluation

Loads a trained `PPO_Flappy` model and runs a single greedy game (argmax over action probabilities). Reports pipe count milestones every 1,000 pipes. Currently loads from `Weights/Old/AT1.pt`.

### `weight_analysis.py` — Visualization of learned activations

Analyzes a trained model across all adaptive layers. Auto-detects the number of layers in the checkpoint. Produces three figures:
1. **Adaptive Layer Overview (Nx3 grid)** — Per-layer basis weight heatmap, combined activation curves for all neurons, and mean basis importance bar charts.
2. **T-alpha Gating (1xN)** — The learned `talpha(z)*z` gate shape for all neurons in each layer.
3. **Basis Functions per Layer (5xN grid)** — Each of the 5 basis functions plotted individually for every neuron in every layer.

### `Extra Scripts/` — Earlier implementations

Older standalone scripts moved here for reference:

| File | Description |
|------|-------------|
| `talpha.py` | Standalone NumPy T-alpha function with timing |
| `custom_network.py` | NumPy network with manual T-alpha backprop |
| `normal_model.py` | PyTorch T-alpha module + comparison vs ReLU |
| `adaptive_model.py` | Adaptive basis layer v1 (mixed bases, `a=100,000`) |
| `test_network.py` | NumPy network evaluation on `y = x^2` |
| `old.py` | Early prototype (superseded) |

## Weights

| File | Architecture | Description |
|------|-------------|-------------|
| `Weights/AT_L5Max.pt` | 5-layer | Latest trained model with per-layer configs |
| `Weights/AT_L4_maxwin.pt` | `[1024, 128, 16, 256, 256]` | Best-performing batched training run |
| `Weights/Old/AT_L5Max.pt` | 5-layer | Earlier 5-layer checkpoint |
| `Weights/Old/AT1.pt` | `[64, 64, 64]` | 3-layer all-Talpha model |
| `Weights/Old/AT_L1.pt` – `AT_L3.pt` | various | Intermediate training checkpoints |
| `Weights/Old/Ad1.pt`, `Ad2.pt` | v1 mixed bases | Original mixed-basis runs |
| `Weights/Old/my_model.npz` | NumPy network | Custom network trained on `y = x^2` |

## Results

| File | Description |
|------|-------------|
| `Results/TA_1024x128x16x256x256.csv` | Training stats (pipes, losses per epoch) for the `[1024, 128, 16, 256, 256]` run |
| `Results/AT_L5/AT_L5 Weight overview.png` | Weight analysis visualization for the 5-layer model |
| `Results/AT_L5/Each function.png` | Individual basis function plots per layer |
| `Results/AT_L5/AT_L5Max.pt` | Copy of the best 5-layer weights |

## File Overview

| File | Purpose |
|------|---------|
| `Talpha.py` | AdaptiveLayer module with STE (shared import) |
| `flappy_agent_Talpha.py` | Modular PPO agent for Flappy Bird |
| `Flappy_Talpha.ipynb` | Batched PPO training notebook |
| `Talpha_no_batch.ipynb` | Single-episode PPO training notebook |
| `train_Talpha.ipynb` | Training via the modular agent |
| `Greedy_Talpha_flappy.py` | Greedy evaluation of a trained model |
| `weight_analysis.py` | Visualization of learned activation shapes |
| `Extra Scripts/` | Earlier standalone implementations |
| `Weights/` | Saved model weights (`.pt`) |
| `Results/` | Training logs and analysis images |
