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

from Talpha import AdaptiveLayer


class PPO_Flappy(nn.Module):
    def __init__(self, hidden_layers, **adaptive_kwargs):
        super().__init__()
        sizes = [8] + hidden_layers
        self.hidden = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.adaptive = nn.ModuleList([
            AdaptiveLayer(size, **adaptive_kwargs) for size in hidden_layers
        ])
        self.actor = nn.Linear(hidden_layers[-1], 2)
        self.critic = nn.Linear(hidden_layers[-1], 1)

    def forward(self, z):
        for linear, adaptive in zip(self.hidden, self.adaptive):
            z = adaptive(linear(z))
        return F.softmax(self.actor(z), dim=-1)


# ── Setup ──

WEIGHT_PATH = "Weights/Old/AT1.pt"
MEANS = torch.tensor([150.0, 0.0, 76.0, 108.0, 208.0, 226.0, 108.0, 208.0])
STDS  = torch.tensor([44.0, 5.0, 44.0, 48.0, 48.0, 44.0, 48.0, 48.0])

model = PPO_Flappy([64, 64, 64])
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
