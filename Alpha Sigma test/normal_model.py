import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class simpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(self.linear2(x))
        output = self.linear3(x)
        return output


""" ----------------------------------------- data --------------------------------------------------------"""

num_features = 1000
x = np.linspace(-5, 5, num_features).astype(np.float32)
y = (x ** 2).astype(np.float32)

x_tensor = torch.from_numpy(x).unsqueeze(1)
y_tensor = torch.from_numpy(y).unsqueeze(1)

""" ----------------------------------------- training ----------------------------------------------------"""

model = simpleNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
epochs = 100

for epoch in range(epochs):
    y_pred = model(x_tensor)
    loss = loss_fn(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

""" ----------------------------------------- plot --------------------------------------------------------"""

with torch.no_grad():
    preds = model(x_tensor).squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Actual (x²)")
plt.plot(x, preds, label="Predicted", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("PyTorch Model Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.show()