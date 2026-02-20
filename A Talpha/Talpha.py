import torch
import torch.nn as nn


class AdaptiveLayer(nn.Module):
    """Per-neuron learnable combination of basis functions.

    Each neuron computes: sum_i( w_i * basis_i(z) )

    Bases: talpha(z), talpha(z)*z², talpha(z)*cos(z), talpha(z)*abs(z), talpha(z)*z, talpha(z)*sin(z)
    """
    BASIS_NAMES = ['Ta(z)', 'Ta(z)*z²', 'Ta(z)*cos(z)', 'Ta(z)*abs(z)', 'Ta(z)*z', 'Ta(z)*sin(z)']
    NUM_BASES = 6

    def __init__(self, features, a=200, a_grad=2, w_init=None, c_init=0.5, d_init=0.0, b_init=1.0):
        super().__init__()
        self.features = features

        # TAlpha params
        self.a = a
        self.a_grad = a_grad
        self.c = nn.Parameter(torch.full((features,), c_init))
        self.d = nn.Parameter(torch.full((features,), d_init))
        self.b = nn.Parameter(torch.full((features,), b_init))

        # Mixing weights per neuron
        if w_init is None:
            w = torch.full((features, self.NUM_BASES), 0.4 / (self.NUM_BASES - 1))
            w[:, 0] = 0.6
        else:
            w = torch.zeros(features, self.NUM_BASES)
            for i, val in enumerate(w_init):
                w[:, i] = val
        self.weights = nn.Parameter(w)

    def _talpha(self, z):
        u_hard = self.a * z / 2 - self.d * self.a
        u_soft = self.a_grad * z / 2 - self.d * self.a_grad

        hard = self.c*(torch.tanh(u_hard)-1) + self.b
        soft = self.c * (torch.tanh(u_soft) - 1) + self.b

        return soft + (hard - soft).detach()

    def forward(self, z):
        ta = self._talpha(z)
        bases = torch.stack([
            ta,                           # Talpha
            ta * z ** 2,                   # Talpha Quadratic
            ta * torch.cos(z),            # Talpha cosine
            ta * torch.abs(z),            # Talpha abs
            ta * z,                       # Talpha Linear
        ], dim=-1)                    # (batch, features, 6)

        return (bases * self.weights).sum(dim=-1)  # (batch, features)
