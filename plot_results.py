import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 8,
    'lines.linewidth': 1,
    'axes.grid': True,
    'grid.color': 'black',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

results_dir = os.path.join(os.path.dirname(__file__), "Results")
plots_dir = os.path.join(os.path.dirname(__file__), "Plots")
os.makedirs(plots_dir, exist_ok=True)

for filename in os.listdir(results_dir):
    if not filename.endswith(".csv"):
        continue
    name = os.path.splitext(filename)[0]
    df = pd.read_csv(os.path.join(results_dir, filename))
    epochs = range(1, len(df) + 1)

    loss_cols = [c for c in df.columns if c.startswith("epoch_loss")]

    # Plot losses
    fig, ax = plt.subplots(figsize=(3.375, 3.375 * 0.6))
    loss_labels = {"clip": r"$L^{\mathrm{CLIP}}$", "val": r"$L^{\mathrm{VF}}$",
                    "ent": r"$L^{S}$", "tot": r"$L^{\mathrm{PPO}}$"}
    for col in loss_cols:
        key = col.replace("epoch_loss_", "")
        ax.plot(epochs, df[col], label=loss_labels.get(key, key))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(ncol=2, loc="upper right")
    # fig.savefig(os.path.join(plots_dir, f"{name}_losses.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, f"{name}_losses.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Plot pipes
    fig, ax = plt.subplots(figsize=(3.375, 3.375 * 0.6))
    ax.plot(epochs, df["epoch_pipes"], "-r", label="Non-greedy policy")
    mask = df["epoch_test_pipes"] >= 0
    ax.plot([e for e, m in zip(epochs, mask) if m],
            df.loc[mask, "epoch_test_pipes"], "+b", label="Greedy policy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of pipes passed")
    ax.set_yscale('log')
    ax.legend()
    # fig.savefig(os.path.join(plots_dir, f"{name}_pipes.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, f"{name}_pipes.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

print("Done. Plots saved to", plots_dir)
