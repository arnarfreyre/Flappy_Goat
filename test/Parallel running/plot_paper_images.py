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

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data")
out_dir = os.path.join(script_dir, "paper images")
os.makedirs(out_dir, exist_ok=True)

# Pick one run per model for 6 models spanning the range
targets = [
    ("NN32x3",   "NN32x3_Run1.csv"),
    ("NN64x3",   "NN64x3_Run4.csv"),
    ("NN128x3",  "NN128x3_Run1.csv"),
    ("NN256x3",  "NN256x3_Run1.csv"),
    ("NN512x3",  "NN512x3_Run1.csv"),
    ("NN1024x3", "NN1024x3_Run3.csv"),
]

for model_name, csv_name in targets:
    csv_path = os.path.join(data_dir, model_name, csv_name)
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_name} — not found")
        continue

    df = pd.read_csv(csv_path)
    epochs = range(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(3.375, 3.375 * 0.75))

    # Non-greedy policy (red line)
    ax.plot(epochs, df["epoch_pipes"], "-r", linewidth=1, label="Non-greedy policy")

    # Greedy policy (blue + markers)
    mask = df["epoch_test_pipes"] >= 0
    ax.plot([e for e, m in zip(epochs, mask) if m],
            df.loc[mask, "epoch_test_pipes"], "+b", markersize=5, label="Greedy policy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of pipes passed")
    ax.set_yscale('log')
    ax.legend()

    label = model_name.replace("NN", "NN").replace("x", "-")
    fig.savefig(os.path.join(out_dir, f"{model_name}_pipes.pdf"), dpi=600, bbox_inches="tight")
    print(f"Saved {model_name}_pipes.pdf")
    plt.close(fig)

print(f"\nDone. Images saved to: {out_dir}")
