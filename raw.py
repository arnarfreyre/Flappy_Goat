import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.fc1 = nn.Linear(1, 4)
        #self.fc2 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


# Each size paired with its own epoch count
sizes =       [4]
epochs_list = [1000000]
"""
sizes =       [4, 10, 250, 500, 1000]
epochs_list = [100000, 100000, 100000, 100000, 100000]
"""
# Data
x = torch.linspace(-10, 10, 10000).unsqueeze(1)
y = torch.exp(x)

final_losses = {}

#fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
fig_approx, ax_approx = plt.subplots(figsize=(10, 6))
ax_approx.plot(x.numpy(), y.numpy(), label="Real x²", linewidth=2, color="black")

for size, num_epochs in zip(sizes, epochs_list):
    print(f"\n--- Training with {size} neurons for {num_epochs} epochs ---")
    model = Network(hidden_size=size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_loss = []
    best_loss = float('inf')
    stale_count = 0
    patience = 500

    for epoch in range(num_epochs):
        pred = model(x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        current_loss = loss.item()
        all_loss.append(current_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if current_loss < best_loss:
            best_loss = current_loss
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= patience:
            print(f"  Early stop at epoch {epoch+1} (no improvement for {patience} epochs) | RMSE: {current_loss:.4f}")
            break

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | RMSE: {current_loss:.4f}")

    final_losses[size] = all_loss[-1]
    #ax_loss.plot(range(len(all_loss)), all_loss, label=f"{size} neurons ({len(all_loss)} ep)")

    with torch.no_grad():
        y_pred = model(x).numpy()
    ax_approx.plot(x.numpy(), y_pred, label=f"{size} neurons", linestyle="--")

ax_approx.set_xlabel("x")
ax_approx.set_ylabel("y")
ax_approx.set_title("Network Approximation of x²")
ax_approx.legend()
fig_approx.tight_layout()
plt.show()

"""
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
bars = ax_bar.bar([str(s) for s in final_losses.keys()], final_losses.values())
ax_bar.bar_label(bars, fmt="%.2f")
ax_bar.set_xlabel("Hidden Layer Size")
ax_bar.set_ylabel("Final RMSE")
ax_bar.set_title("Final Loss by Model Size")
fig_bar.tight_layout()




ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("RMSE")
ax_loss.set_title("Loss Curves by Hidden Layer Size")
ax_loss.legend()
fig_loss.tight_layout()
"""