import os
import re
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

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
csv_files = [f for f in csv_files if os.path.basename(f) != "overview.csv"]

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
    hits = df.index[df["epoch_test_pipes"] >= 100000]
    if len(hits) == 0:
        continue
    first_epoch = hits[0] + 1  # 1-based
    model_sizes.append(neurons)
    epochs_to_100k.append(first_epoch)

fig, ax = plt.subplots(figsize=(7, 3.375 * 0.75))
ax.plot(model_sizes, epochs_to_100k, "o", markersize=5)
ax.set_xlabel("Neurons per layer")
ax.set_ylabel("Epochs to 100 000 pipes")
ax.set_title("Epochs to grokk")
ax.set_xscale('log')

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper images")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "epochs_to_100k.pdf"), dpi=600, bbox_inches="tight")
print("Saved epochs_to_100k.pdf")
plt.close(fig)
