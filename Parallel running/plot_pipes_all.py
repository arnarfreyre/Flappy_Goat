import os
import glob
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

data_dir = os.path.join(os.path.dirname(__file__), "Data")

csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
csv_files = [f for f in csv_files if os.path.basename(f) != "overview.csv"]

fig, ax = plt.subplots(figsize=(7, 4))

for csv_path in csv_files:
    label = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    if "epoch_test_pipes" not in df.columns:
        continue
    epochs = range(1, len(df) + 1)
    mask = df["epoch_test_pipes"] >= 0
    ax.plot([e for e, m in zip(epochs, mask) if m],
            df.loc[mask, "epoch_test_pipes"], ".", markersize=3, label=label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Number of pipes passed")
ax.set_yscale('log')
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(data_dir, "pipes_all.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

# --- Plot 2: Epochs to 100k vs model size ---
import re

model_sizes = []
epochs_to_100k = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if "epoch_test_pipes" not in df.columns:
        continue
    name = os.path.splitext(os.path.basename(csv_path))[0]
    match = re.search(r"NN(\d+)x(\d+)", name)
    if not match:
        continue
    neurons = int(match.group(1))
    layers = int(match.group(2))
    hits = df.index[df["epoch_test_pipes"] >= 100000]
    if len(hits) == 0:
        continue
    first_epoch = hits[0] + 1  # 1-based
    model_sizes.append(neurons)
    epochs_to_100k.append(first_epoch)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(model_sizes, epochs_to_100k, "o", markersize=5)
ax2.set_xlabel("Neurons per layer")
ax2.set_ylabel("Epochs to 100k pipes")
ax2.set_xscale('log')
fig2.tight_layout()
fig2.savefig(os.path.join(data_dir, "epochs_to_100k.pdf"), dpi=600, bbox_inches="tight")
plt.show()
