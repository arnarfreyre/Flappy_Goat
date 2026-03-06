"""Standalone greedy evaluation for trained Flappy Bird PPO agents.

Uses float16 + inference_mode + GreedyHead for maximum throughput.
"""
import argparse
import torch
import torch.nn as nn
import sys
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, '../itml-project2')

from ple.games.flappybird import FlappyBird
from ple import PLE


class GreedyHead(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        sizes = [8] + hidden_layers
        self.hidden = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.actor = nn.Linear(hidden_layers[-1], 2)

    def forward(self, x):
        y = x
        for layer in self.hidden:
            y = torch.tanh(layer(y))
        return self.actor(y)


def load_greedy_head(weights_path, hidden_layers, use_f16):
    """Load training weights into a standalone GreedyHead."""
    state_dict = torch.load(weights_path, weights_only=True)
    # Filter to only hidden + actor keys
    greedy_keys = {}
    for k, v in state_dict.items():
        if k.startswith('hidden.') or k.startswith('actor.'):
            greedy_keys[k] = v
    head = GreedyHead(hidden_layers)
    head.load_state_dict(greedy_keys)
    head.eval()
    if use_f16:
        head.half()
    return head


def run_greedy(head, episode, max_pipes, print_freq, use_f16):
    """Run a single greedy episode and return pipe count."""
    episode.reset_game()
    nr_pipes = 0

    means = torch.tensor([150.0, 0.0, 76.0, 108.0, 208.0, 226.0, 108.0, 208.0])
    stds = torch.tensor([44.0, 5.0, 44.0, 48.0, 48.0, 44.0, 48.0, 48.0])
    if use_f16:
        means = means.half()
        stds = stds.half()

    action_set = episode.getActionSet()
    get_state = episode.getGameState
    game_over = episode.game_over
    act = episode.act
    dtype = torch.float16 if use_f16 else torch.float32

    with torch.inference_mode():
        while not game_over():
            raw = get_state()
            state = (torch.tensor(list(raw.values()), dtype=dtype) - means) / stds
            logits = head(state)
            action = 0 if logits[0] >= logits[1] else 1
            reward = act(action_set[action])
            if reward > 0:
                nr_pipes += 1
                if print_freq and nr_pipes % print_freq == 0:
                    print(f"  {nr_pipes} pipes")
                if max_pipes and nr_pipes >= max_pipes:
                    break
    return nr_pipes


def main():
    parser = argparse.ArgumentParser(description='Greedy evaluation of trained Flappy Bird agent')
    parser.add_argument('--weights', required=True, help='Path to .pt weights file')
    parser.add_argument('--layers', type=int, nargs='+', required=True, help='Hidden layer sizes (e.g. 2048 2048 2048)')
    parser.add_argument('--max-pipes', type=int, default=100000, help='Max pipes per episode (default: 100000)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run (default: 1)')
    parser.add_argument('--print-freq', type=int, default=10000, help='Print every N pipes (default: 10000)')
    parser.add_argument('--no-f16', action='store_true', help='Disable float16 (use float32)')
    args = parser.parse_args()

    use_f16 = not args.no_f16
    print(f"Loading weights from {args.weights}")
    print(f"Network: {args.layers}, dtype: {'float16' if use_f16 else 'float32'}")
    print(f"Max pipes: {args.max_pipes}, Episodes: {args.episodes}")
    print()

    head = load_greedy_head(args.weights, args.layers, use_f16)

    game = FlappyBird()
    episode = PLE(game, fps=30, display_screen=False, force_fps=True)
    episode.init()

    results = []
    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}:")
        pipes = run_greedy(head, episode, args.max_pipes, args.print_freq, use_f16)
        results.append(pipes)
        print(f"  Result: {pipes} pipes\n")

    if len(results) > 1:
        import statistics
        print("--- Summary ---")
        print(f"Mean:   {statistics.mean(results):.1f}")
        print(f"Median: {statistics.median(results):.1f}")
        print(f"Min:    {min(results)}")
        print(f"Max:    {max(results)}")
        if len(results) > 2:
            print(f"Stdev:  {statistics.stdev(results):.1f}")


if __name__ == '__main__':
    main()
