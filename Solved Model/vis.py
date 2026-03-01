import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../itml-project2')
from ple.games.flappybird import FlappyBird
from ple import PLE


class PPO_Flappy(nn.Module):
    def __init__(self, num_layers, layer_specs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_specs[i][0], layer_specs[i][1]))
        self.actor = nn.Linear(layer_specs[-1][1], 2)
        self.critic = nn.Linear(layer_specs[-1][1], 1)

    def forward(self, x):
        for i in self.layers:
            x = F.tanh(i(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_prob, state_value


def normalize_game_state(state):
    means = torch.tensor([256.0, 0.0, 200.0, 200.0, 200.0, 400.0, 200.0, 200.0])
    stds = torch.tensor([128.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    return (state - means) / stds


layer_specs = [[8, 1024], [1024, 128], [128, 16], [16, 256], [256, 256]]
model = PPO_Flappy(5, layer_specs)
model.load_state_dict(torch.load("Parallel 5Mil.pt", map_location="cpu"))
model.eval()

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

with torch.no_grad():
    while True:
        state = normalize_game_state(torch.tensor(list(p.getGameState().values())))
        action_prob, _ = model(state)
        action = torch.argmax(action_prob).item()
        p.act(p.getActionSet()[action])

        if p.game_over():
            p.reset_game()
