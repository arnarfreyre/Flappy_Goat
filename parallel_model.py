import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import sys
import os
from multiprocessing import Pool

os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, 'itml-project2')
# noinspection PyUnresolvedReferences
from ple.games.flappybird import FlappyBird
# noinspection PyUnresolvedReferences
from ple import PLE


class PPO_Flappy(nn.Module):

    def __init__(self,num_layers,layer_specs):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
                self.layers.append(nn.Linear(layer_specs[i][0],layer_specs[i][1]))

        self.actor = nn.Linear(layer_specs[-1][1],2)
        self.critic = nn.Linear(layer_specs[-1][1],1)

    def forward(self,x):

        for i in self.layers:
            x = F.tanh(i(x))

        action_prob = F.softmax(self.actor(x),dim=-1)
        state_value = self.critic(x)
        return action_prob,state_value


def normalize_game_state(state):
    means = torch.tensor([256.0, 0.0, 200.0, 200.0, 200.0, 400.0, 200.0, 200.0])
    stds = torch.tensor([128.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    return (state - means) / stds


def get_Advantage(states, rewards, model, gamma):
    with torch.no_grad():
        _,prev_values = model(states)

    G = 0
    returns = []
    for reward in reversed(rewards):
        G = reward + gamma*G
        G = torch.FloatTensor([G])
        returns.insert(0,G)
    returns = torch.stack(returns)

    A = (returns - prev_values).squeeze()

    return A, returns


def Train_Network(model, optimizer, config, model_index):
    epochs = config["epochs"]
    K_epochs = config["K_epochs"]
    epsilon = config["epsilon"]
    gamma = config["gamma"]
    c0 = config["c0"]
    c1 = config["c1"]
    c2 = config["c2"]

    game = FlappyBird()
    last_print_time = time.time()
    max_reward = -float("inf")

    total_rewards = []
    all_L_clip = []
    all_L_vf = []
    all_L_entropy = []
    stats_rows = []

    for i in range(epochs):
        done = False
        p = PLE(game, fps=30, display_screen=False, force_fps=True)
        p.init()

        log_probs = []
        values = []
        rewards = []
        states = []
        actions = []

        while not done:
            game_state = normalize_game_state(torch.tensor(list(p.getGameState().values())))
            action_prob,critic_val = model(game_state)

            action_prob = action_prob.detach().squeeze()
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()


            if action == 0:
                ret_action = p.getActionSet()[0]
            else:
                ret_action = p.getActionSet()[1]

            reward = p.act(ret_action)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            actions.append(action)
            states.append(game_state)

            if p.game_over():
                done = True
                p.reset_game()

        total_rewards.append(sum(rewards))
        states = torch.stack(states).squeeze()
        log_probs = torch.stack(log_probs).squeeze()
        actions = torch.stack(actions).squeeze()
        A, returns = get_Advantage(states, rewards, model, gamma)

        for j in range(K_epochs):
            action_probs,new_values = model(states)
            new_dist = torch.distributions.Categorical(action_probs)
            new_policy = new_dist.log_prob(actions)

            ratio = torch.exp(new_policy-log_probs)
            clipped_ratio = torch.clamp(ratio,min = 1-epsilon,max= 1+epsilon)

            L_clip = (torch.min(ratio*A,clipped_ratio*A).mean())*c0
            L_vf = ((new_values.squeeze() - returns.squeeze())**2)*c1
            entropy_bonus = (new_dist.entropy())*c2


            #loss = -(c0*L_clip-c1*L_vf+c2*entropy_bonus).mean()
            loss = -(L_clip - L_vf + entropy_bonus).mean()

            all_L_clip.append(L_clip.mean().item())
            all_L_vf.append(L_vf.mean().item())
            all_L_entropy.append(entropy_bonus.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_L_clip = sum(all_L_clip[-K_epochs:]) / K_epochs
        epoch_L_vf = sum(all_L_vf[-K_epochs:]) / K_epochs
        epoch_L_entropy = sum(all_L_entropy[-K_epochs:]) / K_epochs
        stats_rows.append({
            "epoch": i + 1,
            "reward": total_rewards[-1],
            "L_clip": epoch_L_clip,
            "L_vf": epoch_L_vf,
            "L_entropy": epoch_L_entropy,
        })

        if total_rewards[-1] > max_reward:
            max_reward = total_rewards[-1]

        now = time.time()
        if now - last_print_time >= 30:
            pct = int((i + 1) / epochs * 100)
            print(f"  Model {model_index}: {pct}% ({i+1}/{epochs}), max reward: {max_reward:.1f}", flush=True)
            last_print_time = now

    return pd.DataFrame(stats_rows)


def train_one_model(args):
    model_index, layer_specs, config = args
    total_layers = len(layer_specs)

    model = PPO_Flappy(total_layers, layer_specs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    stats = Train_Network(model, optimizer, config, model_index)
    return {
        "Stats": stats,
        "Total layers": total_layers,
        "Layer specs": layer_specs,
    }


def save_run_results(results, model_architectures, config, filepath):
    with open(filepath, "w") as f:
        f.write(f"# Config: lr={config['lr']}, epochs={config['epochs']}, "
                f"K_epochs={config['K_epochs']}, epsilon={config['epsilon']}, "
                f"gamma={config['gamma']}, c0={config['c0']}, c1={config['c1']}, c2={config['c2']}\n")
        for i, arch in enumerate(model_architectures):
            f.write(f"# Model{i}: layers={arch}\n")
        f.write("model,epoch,reward,L_clip,L_vf,L_entropy\n")
        for i, result in enumerate(results):
            df = result["Stats"]
            for _, row in df.iterrows():
                f.write(f"Model{i},{int(row['epoch'])},{row['reward']},{row['L_clip']},{row['L_vf']},{row['L_entropy']}\n")


def run_config():
    l1 = [[8, 1024], [1024, 512], [512, 2048]]
    l2 = [[8, 1024], [1024, 128], [128, 16], [16, 256], [256, 256]]
    l3 = [[8, 64], [64, 128], [128, 128]]
    l4 = [[8, 16], [1024, 1024], [1024, 256]]
    l5 = [[8, 512], [512, 64], [64, 256], [256, 256], [256, 256]]
    l6 = [[8, 16], [16, 8], [8, 16], [16, 8], [8, 16]]
    l7 = [[8, 1024], [1024, 8], [8, 256], [256, 256]]

    model_architectures = [l1, l2, l3, l4, l5, l6, l7]

    lr_list = [0.0001, 0.001, 0.01]
    epochs_list = [2000, 2000, 2000]
    K_epochs_list = [5, 5, 5]
    epsilon_list = [0.1, 0.1, 0.1]
    gamma_list = [0.99, 0.99, 0.99]
    c0_list = [1, 1, 1]
    c1_list = [0.0001, 0.0001, 0.0001]
    c2_list = [0.01, 0.01, 0.01]
    batches = [model_architectures, model_architectures, model_architectures]
    runs_per_config = [3, 3, 3]

    num_configs = len(lr_list)
    output_dir = "Multi parallel data"
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    file_count = 0

    for cfg_idx in range(num_configs):
        config = {
            "lr": lr_list[cfg_idx],
            "epochs": epochs_list[cfg_idx],
            "K_epochs": K_epochs_list[cfg_idx],
            "epsilon": epsilon_list[cfg_idx],
            "gamma": gamma_list[cfg_idx],
            "c0": c0_list[cfg_idx],
            "c1": c1_list[cfg_idx],
            "c2": c2_list[cfg_idx],
        }
        architectures = batches[cfg_idx]

        for run_idx in range(runs_per_config[cfg_idx]):
            run_start = time.time()
            print(f"Config {cfg_idx} (lr={config['lr']}), Run {run_idx} — training {len(architectures)} models...", flush=True)

            jobs = [(i, arch, config) for i, arch in enumerate(architectures)]

            with Pool(len(architectures)) as pool:
                results = pool.map(train_one_model, jobs)

            elapsed = time.time() - run_start
            filepath = os.path.join(output_dir, f"config{cfg_idx}_run{run_idx}.txt")
            save_run_results(results, architectures, config, filepath)
            file_count += 1
            print(f"  Saved {filepath} ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\nAll done. {file_count} files saved to '{output_dir}/' in {total_elapsed:.1f}s")


if __name__ == "__main__":
    run_config()
