import os
import pandas as pd
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots()
    for col in loss_cols:
        label = col.replace("epoch_loss_", "")
        ax.plot(epochs, df[col], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} - Losses")
    ax.legend()
    fig.savefig(os.path.join(plots_dir, f"{name}_losses.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot pipes
    fig, ax = plt.subplots()
    ax.plot(epochs, df["epoch_pipes"], label="epoch_pipes")
    mask = df["epoch_test_pipes"] >= 0
    ax.plot([e for e, m in zip(epochs, mask) if m],
            df.loc[mask, "epoch_test_pipes"], "o", label="epoch_test_pipes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pipes")
    ax.set_title(f"{name} - Pipes")
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(plots_dir, f"{name}_pipes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

print("Done. Plots saved to", plots_dir)
