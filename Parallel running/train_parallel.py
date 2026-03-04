"""Parallel training of all Flappy Bird PPO model architectures.

Trains 7 model sizes (64x3 through 4096x3), 3 independent runs each,
using multiprocessing.Pool. Each run trains until 3 consecutive greedy
evaluations hit the 100K pipe cap.

Usage:
    python train_parallel.py                  # use all CPUs
    python train_parallel.py --workers 4      # use 4 workers
"""

import os
import re
import sys
import time
import warnings
from collections import deque
from multiprocessing import Pool, Lock

os.environ['SDL_VIDEODRIVER'] = 'dummy'
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'itml-project2'))

from flappy_agent import FlappyAgent
from ple.games.flappybird import FlappyBird
from ple import PLE
from silence_libpng import patch_flappy
patch_flappy(FlappyBird)

from models_config import models, model_names

# Training hyperparameters (matching notebook values)
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
CLIP_COEF = 1.0
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 5.0
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
TARGET_STEPS = 8192
MINIBATCH_SIZE = 512
EMA_ALPHA = 0.9
VALUE_LOSS = 'mse'
NORMALIZE_ADVANTAGE = True
REWARD_VALUES = {'positive': 1.0, 'tick': 0.0, 'loss': -5.0}
LR_ANNEAL_HORIZON = 5000  # matches notebook num_epochs for identical LR schedule

MAX_PIPES = 100000
CONVERGENCE_STREAK = 3
RUNS_PER_MODEL = 1
TOTAL_WORKERS = 4

# Shared lock for writing to overview.csv from multiple workers
_overview_lock = Lock()


def _next_run_index(csv_dir, model_name):
    """Find the next available Run index in the model's data directory."""
    existing = []
    if os.path.isdir(csv_dir):
        for fname in os.listdir(csv_dir):
            m = re.match(rf'^{re.escape(model_name)}_Run(\d+)\.csv$', fname)
            if m:
                existing.append(int(m.group(1)))
    return max(existing, default=0) + 1


def train_one_run(args):
    """Worker function: train a single model/run combo until convergence."""
    import torch

    layers, model_name, _ = args

    # Setup agent
    agent = FlappyAgent(layers)
    agent.optimizer = torch.optim.Adam(
        agent.network.parameters(), lr=LEARNING_RATE
    )

    # Setup game
    game = FlappyBird()
    episode = PLE(game, fps=30, display_screen=False, force_fps=True,
                  reward_values=REWARD_VALUES)
    episode.init()
    agent.prepare_greedy()

    # Setup CSV output — pick next available Run index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, 'Data', model_name)
    os.makedirs(csv_dir, exist_ok=True)
    run_idx = _next_run_index(csv_dir, model_name)
    tag = f"[{model_name} Run{run_idx}]"
    csv_path = os.path.join(csv_dir, f'{model_name}_Run{run_idx}.csv')

    with open(csv_path, 'w') as f:
        f.write('epoch_pipes,epoch_loss_clip,epoch_loss_val,epoch_loss_ent,epoch_loss_tot,epoch_test_pipes\n')

    epoch = 0
    max_explore = 0.0
    max_greedy = 0
    recent_explore = deque(maxlen=10)
    recent_greedy = deque(maxlen=10)
    last_print_time = time.time()
    converged = False
    test_freq = 1
    test_threshold = 10

    while not converged:
        # Anneal learning rate (matching run_training() schedule)
        lr = LEARNING_RATE * (1.0 - epoch / LR_ANNEAL_HORIZON)
        lr = max(lr, 0.0)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr

        # Collect training data
        batch = []
        episode_pipes = []
        total_steps = 0
        while total_steps < TARGET_STEPS:
            nr_pipes = agent.play_episode(episode, mode='Explore')
            batch.append(agent.memory)
            episode_pipes.append(nr_pipes)
            total_steps += agent.memory.shape[0]
        avg_pipes = sum(episode_pipes) / len(episode_pipes)

        # Train
        l_clip, l_vf, l_ent, loss = agent.train(
            batch, GAMMA, LAM, CLIP_EPS, CLIP_COEF, VALUE_COEF,
            ENTROPY_COEF, MAX_GRAD_NORM, PPO_EPOCHS, MINIBATCH_SIZE,
            EMA_ALPHA, VALUE_LOSS, NORMALIZE_ADVANTAGE
        )

        # Track stats
        max_explore = max(max_explore, avg_pipes)
        recent_explore.append(avg_pipes)

        # Test the greedy policy (adaptive frequency, matching run_training())
        test_pipes = -1
        if epoch % test_freq == 0:
            test_pipes = agent.play_greedy(episode, max_pipes=MAX_PIPES)
            max_greedy = max(max_greedy, test_pipes)
            recent_greedy.append(test_pipes)
            if test_pipes >= test_threshold:
                test_freq = min(test_freq * 2, 16)
                test_threshold = min(test_threshold * 10, 100000)
            elif test_pipes < test_threshold / 10:
                test_freq = max(test_freq / 2, 1)
                test_threshold = max(test_threshold / 10, 10)

        # Append to CSV (incremental — survives kills)
        with open(csv_path, 'a') as f:
            f.write(f'{avg_pipes},{l_clip},{l_vf},{l_ent},{loss},{test_pipes}\n')

        now = time.time()
        if now - last_print_time >= 45:
            avg10_explore = sum(recent_explore) / len(recent_explore)
            avg10_greedy = sum(recent_greedy) / len(recent_greedy) if recent_greedy else 0
            print(f"{tag} Epoch {epoch} | "
                  f"explore: last={avg_pipes:.1f} max={max_explore:.1f} avg10={avg10_explore:.1f} | "
                  f"greedy: last={test_pipes} max={max_greedy} avg10={avg10_greedy:.0f} | "
                  f"streak={1 if test_pipes >= MAX_PIPES else 0}/{CONVERGENCE_STREAK}",
                  flush=True)
            last_print_time = now

        epoch += 1

        # Greedy-only convergence check
        if test_pipes >= MAX_PIPES:
            streak = 1
            print(f"{tag} Hit {MAX_PIPES} pipes — running {CONVERGENCE_STREAK - 1} "
                  f"greedy-only confirmation evals...", flush=True)
            for i in range(CONVERGENCE_STREAK - 1):
                extra_greedy = agent.play_greedy(episode, max_pipes=MAX_PIPES)
                max_greedy = max(max_greedy, extra_greedy)
                recent_greedy.append(extra_greedy)
                # Log greedy-only eval with 0s for training columns
                with open(csv_path, 'a') as f:
                    f.write(f'0,0,0,0,0,{extra_greedy}\n')
                if extra_greedy < MAX_PIPES:
                    print(f"{tag} Greedy-only eval {i + 1}: {extra_greedy} pipes "
                          f"— streak broken, resuming training", flush=True)
                    break
                streak += 1
                print(f"{tag} Greedy-only eval {i + 1}: {extra_greedy} pipes "
                      f"(streak={streak}/{CONVERGENCE_STREAK})", flush=True)
            if streak == CONVERGENCE_STREAK:
                converged = True

    print(f"{tag} CONVERGED at epoch {epoch} "
          f"({CONVERGENCE_STREAK}x consecutive {MAX_PIPES} pipes)", flush=True)

    # Log completion to overview.csv
    overview_path = os.path.join(script_dir, 'Data', 'overview.csv')
    with _overview_lock:
        write_header = not os.path.exists(overview_path) or os.path.getsize(overview_path) == 0
        with open(overview_path, 'a') as f:
            if write_header:
                f.write('model,run,epochs,max_explore,max_greedy,csv_file\n')
            f.write(f'{model_name},{run_idx},{epoch},{max_explore:.1f},{max_greedy},'
                    f'{model_name}_Run{run_idx}.csv\n')


def main():
    # Build job list: every model × RUNS_PER_MODEL runs
    jobs = []
    for layers, name in zip(models, model_names):
        for i in range(RUNS_PER_MODEL):
            jobs.append((layers, name, i))  # i is unused; run_idx determined at runtime

    print(f"Launching {len(jobs)} training jobs with {TOTAL_WORKERS} workers")
    print(f"Models: {model_names}")
    print(f"Runs per model: {RUNS_PER_MODEL}")
    print(f"Convergence: {CONVERGENCE_STREAK}x consecutive {MAX_PIPES} pipes")
    print(f"CSV output: Parallel running/Data/<model>/<model>_Run<n>.csv")
    print(f"Overview:   Parallel running/Data/overview.csv")
    print()

    with Pool(processes=TOTAL_WORKERS) as pool:
        pool.map(train_one_run, jobs)

    print("\nAll jobs complete.")


if __name__ == '__main__':
    main()
