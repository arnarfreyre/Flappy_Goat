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

# Hyperparameters
lr = 0.0001
epsilon = 0.1
gamma = 0.99
c0 = 1
c1 = 0.01
c2 = 0.1
epochs = 3000
K_epochs = 5
Total_random_runs = 10


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


def get_Advantage(states, rewards, model):
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


def get_children():
    total_layers = 5
    input_layer = 8
    layer_specs = []
    layer_setup = [[8, 512], [512, 128], [128, 8], [8, 256], [256, 512]]

    layer_specs.append([input_layer,int(layer_setup[0][1]*2**(random.randint(-1,1)))])
    for i in range(total_layers-1):
        layer_specs.append([layer_specs[i][1],int(layer_setup[i+1][1]*2**(random.randint(-1,1)))])
    return total_layers,layer_specs


def get_random_layers():
    input_layer = 8
    total_layers = random.randint(1, 8)
    layer_specs = []
    layer_specs.append([input_layer, 2 ** (3 + random.randint(1, 8))])

    for i in range(total_layers - 1):
        layer_specs.append([layer_specs[i][1], int(layer_specs[i][1]*2 ** (random.randint(-2, 2)))])

    total_size = 0
    for i in layer_specs:
        size = i[0]*i[1] + i[1]
        total_size += size

    print(f"Total Network Size: {total_size}")

    return total_layers, layer_specs

def determined_network():
    total_layers = 5
    layer_specs = [[8, 1024], [1024, 128], [128, 16], [16, 256], [256, 256]]


    return total_layers,layer_specs

def Train_Network(model, optimizer, epochs):
    print_freq = 100

    start_time = time.time()
    game = FlappyBird()

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

            critic_val = critic_val.detach().squeeze()

            if action == 0:
                ret_action = p.getActionSet()[0]
            else:
                ret_action = p.getActionSet()[1]

            reward = p.act(ret_action)

            log_probs.append(dist.log_prob(action))
            values.append(critic_val)
            rewards.append(reward)
            actions.append(action)
            states.append(game_state)

            if p.game_over():
                done = True
                p.reset_game()

        total_rewards.append(sum(rewards))
        states = torch.stack(states).squeeze()
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs).squeeze()
        actions = torch.stack(actions).squeeze()
        A, returns = get_Advantage(states, rewards, model)

        for j in range(K_epochs):
            action_probs,new_values = model(states)
            new_dist = torch.distributions.Categorical(action_probs)
            new_policy = new_dist.log_prob(actions)

            ratio = torch.exp(new_policy-log_probs)
            clipped_ratio = torch.clamp(ratio,min = 1-epsilon,max= 1+epsilon)

            L_clip = (torch.min(ratio*A,clipped_ratio*A).mean())*c0
            L_vf = ((new_values.squeeze() - returns.squeeze())**2)*c1
            entropy_bonus = (new_dist.entropy())*c2

            loss = -(c0*L_clip-c1*L_vf+c2*entropy_bonus).mean()

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

        if (i+1) % print_freq == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            epoch_num = i+1
            n_recent = int(K_epochs*print_freq)
            avg_reward = sum(total_rewards[-print_freq:]) /len(total_rewards[-print_freq:])
            L_clip_mean = sum(all_L_clip[-n_recent:]) / len(all_L_clip[-n_recent:])
            L_vf_mean = sum(all_L_vf[-n_recent:]) / len(all_L_vf[-n_recent:])
            L_entr_mean = sum(all_L_entropy[-n_recent:]) / len(all_L_entropy[-n_recent:])
            print(f"[Model {os.getpid()}] Epoch: {epoch_num}, "
                  f"L_clip: {L_clip_mean:.2f}, "
                  f"Squared loss: {L_vf_mean:.2f}, "
                  f"Entropy loss: {L_entr_mean:.2f}, "
                  f"Avg Rewards: {avg_reward:.2f}, "
                  f"Time elapsed: {elapsed_time:.2f}s")

    return pd.DataFrame(stats_rows)


def train_one_model(model_index):
    total_layers, layer_specs = determined_network()
    print(f"[Model{model_index}] Layers: {total_layers}, Setup: {layer_specs}", flush=True)

    model = PPO_Flappy(total_layers, layer_specs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = Train_Network(model, optimizer, epochs)
    model_str = "flappy weights/Parallel " + str(random.randint(0,100000000))+".pt"
    torch.save(model.state_dict(), model_str)
    return {
        "Stats": stats,
        "Total layers": total_layers,
        "Layer specs": layer_specs,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Starting {Total_random_runs} parallel training runs...")
    start = time.time()

    with Pool(Total_random_runs) as pool:
        results = pool.map(train_one_model, range(Total_random_runs))

    elapsed = time.time() - start
    print(f"\nAll {Total_random_runs} models finished in {elapsed:.1f}s")

    model_dict = {}
    for i, result in enumerate(results):
        model_dict[f"Model{i}"] = result

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for name, data in model_dict.items():
        df = data["Stats"]
        sizes = " -> ".join([str(s[1]) for s in data["Layer specs"]])
        label = f"{name} ({data['Total layers']}L: {sizes})"
        ax1.plot(df["epoch"], df["L_clip"], label=label, alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L_clip Loss")
    ax1.set_title("Clipping Loss per Model")
    ax1.legend(fontsize=7, loc="best")
    ax1.set_xlim(0, epochs)
    ax1.grid(True, alpha=0.3)

    for name, data in model_dict.items():
        df = data["Stats"]
        sizes = " -> ".join([str(s[1]) for s in data["Layer specs"]])
        label = f"{name} ({data['Total layers']}L: {sizes})"
        ax2.plot(df["epoch"], df["reward"].rolling(50, min_periods=1).mean(), label=label, alpha=0.8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg Reward (rolling 50)")
    ax2.set_title("Reward per Model")
    ax2.legend(fontsize=7, loc="best")
    ax2.set_xlim(0, epochs)
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("parallel_results.png", dpi=150)
    print("Plots saved to parallel_results.png")

    # --- Fastest to 0 avg reward ---
    fastest_model = None
    fastest_epoch = float("inf")

    for name, data in model_dict.items():
        df = data["Stats"]
        rolling_reward = df["reward"].rolling(50, min_periods=1).mean()
        reached = rolling_reward[rolling_reward >= 0]
        if len(reached) > 0:
            epoch = df.loc[reached.index[0], "epoch"]
            if epoch < fastest_epoch:
                fastest_epoch = epoch
                fastest_model = name

    if fastest_model:
        d = model_dict[fastest_model]
        print(f"\nFastest to 0 avg reward: {fastest_model} at epoch {fastest_epoch}")
        print(f"  Layers: {d['Total layers']}, Specs: {d['Layer specs']}")
    else:
        print("\nNo model reached 0 avg reward.")

    # --- Best overall score ---
    best_model = None
    best_reward = -float("inf")

    for name, data in model_dict.items():
        max_reward = data["Stats"]["reward"].max()
        if max_reward > best_reward:
            best_reward = max_reward
            best_model = name

    d = model_dict[best_model]
    print(f"\nBest single-epoch reward: {best_model} with reward {best_reward:.2f}")
    print(f"  Layers: {d['Total layers']}, Specs: {d['Layer specs']}")

    # --- Ranking table ---
    rows = []
    for name, data in model_dict.items():
        df = data["Stats"]
        rolling_reward = df["reward"].rolling(50, min_periods=1).mean()
        reached = rolling_reward[rolling_reward >= 0]
        epoch_to_zero = int(df.loc[reached.index[0], "epoch"]) if len(reached) > 0 else None
        sizes = " -> ".join([str(s[1]) for s in data["Layer specs"]])

        rows.append({
            "Model": name,
            "Layers": data["Total layers"],
            "Architecture": f"8 -> {sizes}",
            "Best Reward": f"{df['reward'].max():.1f}",
            "Final Avg Reward (last 100)": f"{df['reward'].tail(100).mean():.1f}",
            "Epoch to 0 Reward": epoch_to_zero if epoch_to_zero else "Never",
        })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("Final Avg Reward (last 100)", ascending=False, key=lambda x: pd.to_numeric(x, errors="coerce")).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "Rank"
    print("\n--- Model Ranking ---")
    print(ranking.to_string())
