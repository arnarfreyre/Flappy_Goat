import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys
sys.path.insert(0, '../itml-project2')
from ple.games.flappybird import FlappyBird
from ple import PLE


# ── Model (inference-only, no critic needed but kept for weight loading) ──

class AdaptiveLayer(nn.Module):
    NUM_BASES = 5

    def __init__(self, features, a=20):
        super().__init__()
        self.features = features
        self.a = a
        self.c = nn.Parameter(torch.full((features,), 0.5))
        self.d = nn.Parameter(torch.zeros(features))
        self.b = nn.Parameter(torch.ones(features))
        w_init = torch.zeros(features, self.NUM_BASES)
        w_init[:, 0] = 1.0
        self.weights = nn.Parameter(w_init)

    def _talpha(self, z):
        u = self.a * z / 2 - self.d * self.a
        return self.c * (torch.tanh(u / 2) - 1) + self.b

    def forward(self, z):
        bases = torch.stack([
            self._talpha(z),                           # Talpha
            self._talpha(z) * z ** 2,                   # Talpha Quadratic
            self._talpha(z) * torch.cos(z),            # Talpha cosine
            self._talpha(z) * torch.sin(z),            # Talpha sine
            self._talpha(z) * z,                       # Talpha Linear
        ], dim=-1)                    # (batch, features, 5)

        return (bases * self.weights).sum(dim=-1)


class PPO_Flappy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.adaptive1 = AdaptiveLayer(64)
        self.fc2 = nn.Linear(64, 64)
        self.adaptive2 = AdaptiveLayer(64)
        self.fc3 = nn.Linear(64, 64)
        self.adaptive3 = AdaptiveLayer(64)
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, z):
        z = self.adaptive1(self.fc1(z))
        z = self.adaptive2(self.fc2(z))
        z = self.adaptive3(self.fc3(z))
        return F.softmax(self.actor(z), dim=-1)


# ── Setup ──

WEIGHT_PATH = "Weights/AT1.pt"
MEANS = torch.tensor([150.0, 0.0, 76.0, 108.0, 208.0, 226.0, 108.0, 208.0])
STDS  = torch.tensor([44.0, 5.0, 44.0, 48.0, 48.0, 44.0, 48.0, 48.0])

model = PPO_Flappy()
model.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))
model.eval()

game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=True,
        reward_values={'positive': 1.0, 'tick': 0.0, 'loss': -5.0})
p.init()
actions = p.getActionSet()

# ── Single greedy run ──

total_pipes = 0
last_report = 0
start = time.time()

p.reset_game()
with torch.no_grad():
    while not p.game_over():
        state = (torch.tensor(list(p.getGameState().values())) - MEANS) / STDS
        probs = model(state)
        action = actions[probs.argmax().item()]
        reward = p.act(action)
        if reward > 0:
            total_pipes += 1
            if total_pipes // 1000 > last_report:
                last_report = total_pipes // 1000
                elapsed = time.time() - start
                print(f"Pipes: {total_pipes:,}  |  Time: {elapsed:.1f}s")

elapsed = time.time() - start
print(f"\nGame over — Total pipes: {total_pipes:,}  |  Time: {elapsed:.1f}s")
