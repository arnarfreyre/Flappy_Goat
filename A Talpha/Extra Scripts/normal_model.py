import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TAlpha(nn.Module):
    """Learnable activation: h(x) = c * (tanh(a*x/2 - d*a) - 1) + b
    Per-neuron learnable params: c, d, b. a is fixed (controls step sharpness)."""
    def __init__(self, features, a=10000):
        super().__init__()
        self.a = a
        self.c = nn.Parameter(torch.full((features,), 0.5))
        self.d = nn.Parameter(torch.zeros(features))
        self.b = nn.Parameter(torch.ones(features))

    def forward(self, x):
        u = self.a * x / 2 - self.d * self.a
        h = self.c * (torch.tanh(u / 2) - 1) + self.b
        return h


class simpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        output = self.linear3(x)
        return output


class talphaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.talpha1 = TAlpha(4)
        self.linear2 = nn.Linear(4,4)
        self.talpha2 = TAlpha(4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        z = self.linear1(x)
        z = self.talpha1(z) * z
        z = self.linear2(z)
        z = self.talpha2(z) * z
        output = self.linear3(z)
        return output


""" ----------------------------------------- data --------------------------------------------------------"""

num_features = 10000
x = np.linspace(-10, 10, num_features).astype(np.float32)
y = (x ** 2).astype(np.float32)

x_tensor = torch.from_numpy(x).unsqueeze(1)
y_tensor = torch.from_numpy(y).unsqueeze(1)

""" ----------------------------------------- training ----------------------------------------------------"""

models = {
    "ReLU network": simpleNetwork(),
    "TAlpha network": talphaNetwork(),
}

results = {}
epochs = 20000
lr = 0.001

for name, model in models.items():
    print(f"\n--- {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        y_pred = model(x_tensor)
        loss = loss_fn(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    with torch.no_grad():
        results[name] = model(x_tensor).squeeze().numpy()

""" ----------------------------------------- plot --------------------------------------------------------"""

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Actual (x²)", linewidth=2)
for name, preds in results.items():
    plt.plot(x, preds, label=name, linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.show()