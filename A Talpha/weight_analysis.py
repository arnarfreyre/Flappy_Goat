"""
Weight analysis for trained PPO Flappy Bird model with AdaptiveLayers.
Visualizes the learned basis weights, combined activation shapes,
and T-alpha gating for all 5 adaptive layers.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────
A = 200  # must match the 'a' used during training (AdaptiveLayer default)
MODEL_PATH = "Weights/AT_2.pt"
X = np.linspace(-5, 5, 500)
ALL_BASIS_LABELS = ['Ta(z)', 'Ta(z)*z²', 'Ta(z)*cos(z)', 'Ta(z)*abs(z)', 'Ta(z)*z', 'Ta(z)*sin(z)']
ALL_BASIS_KEYS   = ['w_Ta',  'w_Ta_z2',  'w_Ta_cos',   'w_Ta_abs',   'w_Ta_z',  'w_Ta_sin']
ALL_LAYER_NAMES = ["adaptive1", "adaptive2", "adaptive3", "adaptive4", "adaptive5"]
# ModuleList variant: adaptive.0, adaptive.1, ...
ALL_LAYER_NAMES_ALT = ["adaptive.0", "adaptive.1", "adaptive.2", "adaptive.3", "adaptive.4"]


# ── Helper functions (pure numpy, vectorized over neurons) ─────────────
def talpha_np(z, c, d, b, a=A):
    """Compute talpha(z) = c * (tanh(u/2) - 1) + b where u = a*z/2 - d*a.
    z: (N,)  c,d,b: (M,)  → returns (M, N)
    """
    z_ = z[np.newaxis, :]       # (1, N)
    c_ = c[:, np.newaxis]       # (M, 1)
    d_ = d[:, np.newaxis]
    b_ = b[:, np.newaxis]
    u = a * z_ / 2 - d_ * a    # (M, N)
    return c_ * (np.tanh(u / 2) - 1) + b_


def all_bases_np(z, c, d, b, a=A):
    """All possible basis functions. Returns list of (M, N) arrays."""
    ta = talpha_np(z, c, d, b, a)
    z_ = z[np.newaxis, :]
    return [
        ta,                                # talpha(z)
        ta * z_ ** 2,                      # talpha(z)*z²
        ta * np.cos(z_),                   # talpha(z)*cos(z)
        ta * np.abs(z_),                   # talpha(z)*abs(z)
        ta * z_,                           # talpha(z)*z
        ta * np.sin(z_),                   # talpha(z)*sin(z)
    ]


def combined_activation_np(z, weights, c, d, b, a=A):
    """Weighted sum of basis functions for each neuron.
    Automatically matches the number of bases to weights.shape[1].
    """
    num_bases = weights.shape[1]
    bases = np.stack(all_bases_np(z, c, d, b, a)[:num_bases], axis=-1)
    w = weights[:, np.newaxis, :]
    return (bases * w).sum(axis=-1)


# ── Load model weights ────────────────────────────────────────────────
sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

# Auto-detect which adaptive layers exist in the checkpoint (handles both naming conventions)
LAYER_NAMES = [n for n in ALL_LAYER_NAMES if f"{n}.c" in sd]
if not LAYER_NAMES:
    LAYER_NAMES = [n for n in ALL_LAYER_NAMES_ALT if f"{n}.c" in sd]
NUM_LAYERS = len(LAYER_NAMES)

layers = []
for name in LAYER_NAMES:
    layers.append({
        "c": sd[f"{name}.c"].numpy(),
        "d": sd[f"{name}.d"].numpy(),
        "b": sd[f"{name}.b"].numpy(),
        "weights": sd[f"{name}.weights"].numpy(),
    })

# Derive number of bases from loaded weights
NUM_BASES = layers[0]["weights"].shape[1]
BASIS_LABELS = ALL_BASIS_LABELS[:NUM_BASES]
BASIS_KEYS   = ALL_BASIS_KEYS[:NUM_BASES]
ALL_COLORS   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#DD8452"]

# ── Figure 1: Adaptive Layer Overview (5×3) ──────────────────────────
fig1, axes1 = plt.subplots(NUM_LAYERS, 3, figsize=(18, 4 * NUM_LAYERS))
fig1.suptitle("Adaptive Layer Overview", fontsize=16, fontweight="bold")

for row, (layer, name) in enumerate(zip(layers, LAYER_NAMES)):
    w = layer["weights"]
    c, d, b = layer["c"], layer["d"], layer["b"]
    M = w.shape[0]

    # Col 1: Basis weight heatmap
    ax = axes1[row, 0]
    vmax = np.abs(w).max()
    im = ax.imshow(w, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(NUM_BASES))
    ax.set_xticklabels(BASIS_LABELS, fontsize=9)
    ax.set_ylabel(f"{name} ({M})\nNeuron index")
    ax.set_title("Basis Weights" if row == 0 else "")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Col 2: Combined activation curves
    ax = axes1[row, 1]
    activations = combined_activation_np(X, w, c, d, b)  # (M, N)
    for i in range(M):
        ax.plot(X, activations[i], color="steelblue", alpha=0.15, linewidth=0.7)
    ax.plot(X, activations.mean(axis=0), color="darkred", linewidth=2, label="mean")
    ax.set_xlim(-10, 10)
    ax.set_ylabel("activation")
    ax.set_title("Combined Activation Curves" if row == 0 else "")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Col 3: Mean |weight| per basis with std error bars
    ax = axes1[row, 2]
    abs_w = np.abs(w)
    means = abs_w.mean(axis=0)
    stds = abs_w.std(axis=0)
    bars = ax.bar(range(NUM_BASES), means, yerr=stds, capsize=4,
                  color=ALL_COLORS[:NUM_BASES],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(NUM_BASES))
    ax.set_xticklabels(BASIS_LABELS, fontsize=9)
    ax.set_ylabel("mean |weight|")
    ax.set_title("Basis Importance" if row == 0 else "")

fig1.tight_layout(rect=[0, 0, 1, 0.95])

# ── Figure 3: All Basis Functions per Layer (bases × N layers) ────────
fig3, axes3 = plt.subplots(NUM_BASES, NUM_LAYERS,
                            figsize=(4 * NUM_LAYERS, 3 * NUM_BASES))
fig3.suptitle("Learned Basis Functions per Layer", fontsize=16, fontweight="bold")

for col, (layer, name) in enumerate(zip(layers, LAYER_NAMES)):
    c, d, b = layer["c"], layer["d"], layer["b"]
    M = c.shape[0]
    basis_curves = all_bases_np(X, c, d, b)[:NUM_BASES]

    for row, (curves, label) in enumerate(zip(basis_curves, BASIS_LABELS)):
        ax = axes3[row, col]
        for i in range(M):
            ax.plot(X, curves[i], color=ALL_COLORS[row], alpha=0.15, linewidth=0.7)
        ax.plot(X, curves.mean(axis=0), color="darkred", linewidth=2, label="mean")
        ax.set_xlim(-10, 10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if row == 0:
            ax.set_title(f"{name} ({M})")
        if col == 0:
            ax.set_ylabel(label)
        if row == len(BASIS_LABELS) - 1:
            ax.set_xlabel("z")
        ax.legend(loc="upper left", fontsize=7)

fig3.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
