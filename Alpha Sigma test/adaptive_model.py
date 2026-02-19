import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TAlpha(nn.Module):
    """Learnable activation: h(x) = c * (tanh(a*x/2 - d*a) - 1) + b"""
    def __init__(self, features, a=100000):
        super().__init__()
        self.a = a
        self.c = nn.Parameter(torch.full((features,), 0.5))
        self.d = nn.Parameter(torch.zeros(features))
        self.b = nn.Parameter(torch.ones(features))

    def forward(self, x):
        u = self.a * x / 2 - self.d * self.a
        h = self.c * (torch.tanh(u / 2) - 1) + self.b
        return h


class AdaptiveLayer(nn.Module):
    """Per-neuron learnable combination of basis functions.

    Each neuron computes: sum_i( w_i * basis_i(z) )

    Bases: z, z², cos(z), sin(z), talpha(z)*z
    """
    BASIS_NAMES = ['z', 'z²', 'cos(z)', 'sin(z)', 'talpha(z)*z']
    NUM_BASES = 5

    def __init__(self, features, a=100000):
        super().__init__()
        self.features = features

        # TAlpha params (for the talpha*z basis)
        self.a = a
        self.c = nn.Parameter(torch.full((features,), 0.5))
        self.d = nn.Parameter(torch.zeros(features))
        self.b = nn.Parameter(torch.ones(features))

        # Mixing weights per neuron — init biased toward linear (identity)
        w_init = torch.zeros(features, self.NUM_BASES)
        w_init[:, 0] = 1.0
        self.weights = nn.Parameter(w_init)

    def _talpha(self, z):
        u = self.a * z / 2 - self.d * self.a
        return self.c * (torch.tanh(u / 2) - 1) + self.b

    def forward(self, z):
        bases = torch.stack([
            z,                        # linear
            z ** 2,                   # quadratic
            torch.cos(z),            # cosine
            torch.sin(z),            # sine
            self._talpha(z) * z,     # learnable step * input
        ], dim=-1)                    # (batch, features, 5)

        return (bases * self.weights).sum(dim=-1)  # (batch, features)


""" --------------------------------------------- networks ---------------------------------------------------"""


class simpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


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


class adaptiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.adaptive1 = AdaptiveLayer(4)
        self.linear2 = nn.Linear(4, 4)
        self.adaptive2 = AdaptiveLayer(4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        z = self.linear1(x)
        z = self.adaptive1(z)
        z = self.linear2(z)
        z = self.adaptive2(z)
        output = self.linear3(z)
        return output


if __name__ == "__main__":

    """ ----------------------------------------- data --------------------------------------------------------"""

    num_features = 10000
    x = np.linspace(-15, 15, num_features).astype(np.float32) + np.random.randn(num_features).astype(np.float32)*10
    y = (x ** 2).astype(np.float32)

    x_tensor = torch.from_numpy(x).unsqueeze(1)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    """ ----------------------------------------- training ----------------------------------------------------"""

    models = {
        "ReLU network": simpleNetwork(),
        "TAlpha network": talphaNetwork(),
        "Adaptive network": adaptiveNetwork(),
    }

    results = {}
    epochs = 10000

    for name, model in models.items():
        print(f"\n--- {name} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

    """ ----------------------------------------- plot predictions ---------------------------------------------"""

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Actual (x²)", linewidth=2)
    for name, preds in results.items():
        plt.plot(x, preds, label=name, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Predictions vs Actual")
    plt.legend()
    plt.grid(True)

    """ ----------------------------------------- plot basis weights --------------------------------------------"""

    adaptive_model = models["Adaptive network"]
    learned = adaptive_model.adaptive1.weights.detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(learned, aspect='auto', cmap='RdBu_r')
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Neuron {i}" for i in range(4)])
    ax.set_xticks(range(AdaptiveLayer.NUM_BASES))
    ax.set_xticklabels(AdaptiveLayer.BASIS_NAMES)
    ax.set_title("Learned Basis Mixing Weights")
    for i in range(4):
        for j in range(AdaptiveLayer.NUM_BASES):
            ax.text(j, i, f"{learned[i, j]:.2f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    plt.show()
