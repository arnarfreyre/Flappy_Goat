import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Parse the log file
data = defaultdict(lambda: {"epochs": [], "l_clip": [], "squared_loss": [], "entropy_loss": [], "avg_rewards": []})

with open("Parallel run 1.txt", "r") as f:
    for line in f:
        match = re.match(
            r"\[Model (\d+)\] Epoch: (\d+), L_clip: ([-\d.]+), Squared loss: ([-\d.]+), Entropy loss: ([-\d.]+), Avg Rewards: ([-\d.]+)",
            line.strip()
        )
        if match:
            model_id = match.group(1)
            data[model_id]["epochs"].append(int(match.group(2)))
            data[model_id]["l_clip"].append(float(match.group(3)))
            data[model_id]["squared_loss"].append(float(match.group(4)))
            data[model_id]["entropy_loss"].append(float(match.group(5)))
            data[model_id]["avg_rewards"].append(float(match.group(6)))

# Sort each model's data by epoch
for model_id in data:
    sorted_indices = sorted(range(len(data[model_id]["epochs"])), key=lambda i: data[model_id]["epochs"][i])
    for key in ["epochs", "l_clip", "squared_loss", "entropy_loss", "avg_rewards"]:
        data[model_id][key] = [data[model_id][key][i] for i in sorted_indices]

# Create friendly labels (sort by model PID for consistent ordering)
sorted_ids = sorted(data.keys(), key=lambda x: int(x))
labels = {mid: f"Model {i+1} ({mid})" for i, mid in enumerate(sorted_ids)}

# Compute average across all models at each common epoch
all_epochs = sorted(set(e for mid in sorted_ids for e in data[mid]["epochs"]))
avg_data = {"epochs": [], "l_clip": [], "squared_loss": [], "entropy_loss": [], "avg_rewards": []}
for epoch in all_epochs:
    vals = {k: [] for k in ["l_clip", "squared_loss", "entropy_loss", "avg_rewards"]}
    for mid in sorted_ids:
        if epoch in data[mid]["epochs"]:
            idx = data[mid]["epochs"].index(epoch)
            for k in vals:
                vals[k].append(data[mid][k][idx])
    avg_data["epochs"].append(epoch)
    for k in vals:
        avg_data[k].append(sum(vals[k]) / len(vals[k]))

# Plot 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PPO Parallel Run 2 - Training Metrics", fontsize=16, fontweight="bold")

metrics = [
    ("l_clip", "L_clip", axes[0, 0]),
    ("squared_loss", "Squared Loss", axes[0, 1]),
    ("entropy_loss", "Entropy Loss", axes[1, 0]),
    ("avg_rewards", "Avg Rewards", axes[1, 1]),
]

for key, title, ax in metrics:
    for model_id in sorted_ids:
        ax.plot(data[model_id]["epochs"], data[model_id][key], label=labels[model_id], alpha=0.5)
    ax.plot(avg_data["epochs"], avg_data[key], label="Average", color="black", linewidth=2.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
