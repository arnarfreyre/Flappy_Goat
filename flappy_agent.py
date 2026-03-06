from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, '../itml-project2')

from ple.games.flappybird import FlappyBird
from ple import PLE


# Helper functions
def normalize_game_state(state):
    means = torch.tensor([  150.0,  0.0,    76.0,   108.0,  208.0,  226.0,  108.0,  208.0])
    stds = torch.tensor([   44.0,   5.0,    44.0,   48.0,   48.0,   44.0,   48.0,   48.0])
    return (state - means) / stds

def state_to_tensor(state):
    tensor_state = torch.tensor(list(state.values()), dtype=torch.float32)
    tensor_state = normalize_game_state(tensor_state)

    return tensor_state

class FlappyNetwork(nn.Module):
    def __init__(self, hidden_layers, initial_weights=None):
        super(FlappyNetwork, self).__init__()
        # Hidden layers
        sizes = [8] + hidden_layers
        self.hidden = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        # Output layers
        self.actor = nn.Linear(hidden_layers[-1], 2)
        self.value = nn.Linear(hidden_layers[-1], 1)
        # Set weights
        if initial_weights is not None:
            self.load_state_dict(initial_weights)

    def forward(self, x):
        # Forward through hidden layers
        y = x
        for layer in self.hidden:
            y = F.tanh(layer(y))
        # Forward through output layers
        probs = F.softmax(self.actor(y), dim=-1)
        value = self.value(y).squeeze(-1)

        return probs, value

class GreedyHead(nn.Module):
    """Minimal forward-only module for greedy evaluation.
    Skips softmax (monotonic, argmax-invariant) and value head (unused).
    Shares weight references with the training network.
    """
    def __init__(self, network):
        super().__init__()
        self.hidden = network.hidden  # shared reference
        self.actor = network.actor    # shared reference

    def forward(self, x):
        y = x
        for layer in self.hidden:
            y = torch.tanh(layer(y))
        return self.actor(y)  # raw logits, skip softmax + value

class FlappyAgent():
    def __init__(self, hidden_layers, weights_path=None):
        # Neural network
        self.network = FlappyNetwork(hidden_layers)
        # Experience storage
        self.memory = None  # Single game path tensor
        # Load weights if provided
        if weights_path is not None:
            self.network.load_state_dict(torch.load(weights_path, weights_only=True))
        self.optimizer = None
        # Value normalization statistics
        self.value_mean = 0.0
        self.value_std = 1.0

    # Gaming functions
    def get_action(self, state, mode='Explore'):
        # Fetch predictions
        probs, value = self.network.forward(state)
        # Convert to distribution
        distribution = torch.distributions.Categorical(probs)
        # Get action
        if mode == 'Explore': # Sample from distribution
            action = distribution.sample()
        elif mode =='Exploit': # Choose most probable value
            action = torch.argmax(probs, dim=-1)
        else: # Something has gone wrong
            print('How did you manage to mess up the mode?? Defaulting to Explore I guess...')
            action = distribution.sample()
        # Get the log-probability
        logp = distribution.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def play_episode(self, episode, mode='Explore', max_pipes=None, print_freq=None):
        episode.reset_game()
        nr_pipes = 0

        # Fast path for greedy testing — no autograd, no distribution, no memory storage
        if mode == 'Exploit':
            action_set = episode.getActionSet()
            with torch.no_grad():
                while not episode.game_over():
                    state = state_to_tensor(episode.getGameState())
                    probs, _ = self.network.forward(state)
                    action = torch.argmax(probs).item()
                    reward = episode.act(action_set[action])
                    if reward > 0:
                        nr_pipes += 1
                        if print_freq is not None and nr_pipes % print_freq == 0:
                            print()
                            print(nr_pipes)
                        if max_pipes is not None and nr_pipes >= max_pipes:
                            break
            return nr_pipes

        # Explore path — full experience collection for training
        game_path = []
        while not episode.game_over():
            state = state_to_tensor(episode.getGameState())
            action, logp, value = self.get_action(state, mode)
            reward = episode.act(episode.getActionSet()[action])
            game_path.append(torch.cat([
                state,
                torch.tensor([float(action), float(reward), float(logp), float(value)],
                             dtype=torch.float32)
                ]))
            if reward > 0:
                nr_pipes += 1
                if print_freq is not None and nr_pipes % print_freq == 0:
                    print()
                    print(nr_pipes)
                if max_pipes is not None and nr_pipes >= max_pipes:
                    break

        self.memory = torch.stack(game_path)
        return nr_pipes

    def prepare_greedy(self):
        """One-time setup for fast greedy evaluation."""
        self._greedy_head = GreedyHead(self.network)
        self._greedy_means = torch.tensor([150.0, 0.0, 76.0, 108.0, 208.0, 226.0, 108.0, 208.0])
        self._greedy_stds = torch.tensor([44.0, 5.0, 44.0, 48.0, 48.0, 44.0, 48.0, 48.0])

    def play_greedy(self, episode, max_pipes=None, print_freq=None):
        """Optimized greedy evaluation using inference_mode and GreedyHead."""
        episode.reset_game()
        nr_pipes = 0

        # Local refs for speed
        head = self._greedy_head
        means = self._greedy_means
        stds = self._greedy_stds
        action_set = episode.getActionSet()
        get_state = episode.getGameState
        game_over = episode.game_over
        act = episode.act

        head.eval()
        with torch.inference_mode():
            while not game_over():
                raw = get_state()
                state = (torch.tensor(list(raw.values()), dtype=torch.float32) - means) / stds
                logits = head(state)
                action = 0 if logits[0] >= logits[1] else 1
                reward = act(action_set[action])
                if reward > 0:
                    nr_pipes += 1
                    if print_freq is not None and nr_pipes % print_freq == 0:
                        print()
                        print(nr_pipes)
                    if max_pipes is not None and nr_pipes >= max_pipes:
                        break
        self.network.train()
        return nr_pipes

    # Training functions
    def run_training(self, gamma, lam, clip_eps, clip_coef, value_coef, entropy_coef, max_grad_norm,
                     learning_rate, ppo_epochs, num_epochs, target_steps=800, minibatch_size=128, print_freq=100,
                     ema_alpha=0.9, value_loss='mse', reward_values=None, normalize_advantage=True, test_exploit=False,
                     result_path=None):
        if reward_values is None:
            shift = 5
        else:
            shift = - reward_values['loss']
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # Start the game
        game = FlappyBird()
        ple_kwargs = dict(fps=30, display_screen=False, force_fps=True)
        if reward_values is not None:
            ple_kwargs['reward_values'] = reward_values
        episode = PLE(game, **ple_kwargs)
        episode.init()
        self.prepare_greedy()

        # Initialize collectors
        epoch_pipes = []
        epoch_loss_clip = []
        epoch_loss_val = []
        epoch_loss_ent = []
        epoch_loss_tot = []
        epoch_test_pipes = []
        test_freq = 1
        test_threshold = 10
        for epoch in range(num_epochs):

            # Anneal learning rate
            lr = learning_rate * (1.0 - epoch / num_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Collect games until target timesteps reached
            batch = []
            episode_pipes = []
            total_steps = 0
            while total_steps < target_steps:
                nr_pipes = self.play_episode(episode, mode='Explore')
                batch.append(self.memory)
                episode_pipes.append(nr_pipes)
                total_steps += self.memory.shape[0]
            avg_pipes = sum(episode_pipes) / len(episode_pipes)

            # Train on the full batch, then discard
            l_clip, l_vf, l_ent, loss = self.train(batch, gamma, lam, clip_eps, clip_coef, value_coef,
                                                    entropy_coef, max_grad_norm, ppo_epochs, minibatch_size,
                                                    ema_alpha, value_loss, normalize_advantage)

            # Test the greedy policy
            test_pipes = -1
            if test_exploit and (epoch % test_freq == 0):
                test_pipes = self.play_greedy(episode, max_pipes=100000, print_freq=10000)
                if test_pipes >= test_threshold:
                    test_freq = min(test_freq*2,16)
                    test_threshold = min(test_threshold*10,100000)
                elif test_pipes < test_threshold/10:
                    test_freq = max(test_freq/2, 1)
                    test_threshold = max(test_threshold/10,10)

            # Collect stats
            epoch_pipes.append(avg_pipes)
            epoch_loss_clip.append(l_clip)
            epoch_loss_val.append(l_vf)
            epoch_loss_ent.append(l_ent)
            epoch_loss_tot.append(loss)
            epoch_test_pipes.append(test_pipes)

            # Save stats
            if result_path is not None:
                with open(result_path, 'w' if epoch == 0 else 'a') as f:
                    if epoch == 0:
                        f.write('epoch_pipes,epoch_loss_clip,epoch_loss_val,epoch_loss_ent,epoch_loss_tot,epoch_test_pipes\n')
                    f.write(f'{epoch_pipes[-1]},{epoch_loss_clip[-1]},{epoch_loss_val[-1]},{epoch_loss_ent[-1]},{epoch_loss_tot[-1]},{epoch_test_pipes[-1]}\n')

            # Print stats
            # if (epoch + 1) % print_freq == 0:

            #     recent_pipes = epoch_pipes[-print_freq:]
            #     recent_loss_clip = epoch_loss_clip[-print_freq:]
            #     recent_loss_val = epoch_loss_val[-print_freq:]
            #     recent_loss_ent = epoch_loss_ent[-print_freq:]
            #     recent_loss_tot = epoch_loss_tot[-print_freq:]
            #     recent_epoch_test_pipes = [p if p > 0 else 0 for p in epoch_test_pipes[-print_freq:]]

            #     print(f"Epoch {epoch+1:5d} | "
            #           f"Avg Pipes: {sum(recent_pipes) / len(recent_pipes):7.2f} | "
            #           f"L_clip: {sum(recent_loss_clip) / len(recent_loss_clip):.4f} | "
            #           f"L_vf: {sum(recent_loss_val) / len(recent_loss_val):.4f} | "
            #           f"L_ent: {sum(recent_loss_ent) / len(recent_loss_ent):.4f} | "
            #           f"Loss: {sum(recent_loss_tot) / len(recent_loss_tot):.4f} | "
            #           f"Test Pipes: {sum(recent_epoch_test_pipes) / len(recent_epoch_test_pipes):7.2f}")

    def train(self, batch, gamma, lam, clip_eps, clip_coef, value_coef, entropy_coef, max_grad_norm, ppo_epochs, minibatch_size,
              ema_alpha=0.9, value_loss='mse', normalize_advantage=True):

        # Pool all episodes from the batch
        all_states, all_actions, all_logp_old, all_rewards = [], [], [], []
        for ep in batch:
            all_states.append(ep[:, 0:8])
            all_actions.append(ep[:, 8].long())
            all_logp_old.append(ep[:, 10])
            all_rewards.append(ep[:, 9])
        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        logp_old = torch.cat(all_logp_old)

        # Use stored V(s) from rollout
        all_values_old = []
        for ep in batch:
            all_values_old.append(ep[:, 11])
        fresh_values = torch.cat(all_values_old)

        # Compute advantages per-episode
        all_adv, all_vtarg = [], []
        offset = 0
        for ep in batch:
            ep_len = ep.shape[0]
            adv, vtarg = self.compute_advantage(all_rewards[len(all_adv)],
                                                fresh_values[offset:offset + ep_len], gamma, lam)
            all_adv.append(adv)
            all_vtarg.append(vtarg)
            offset += ep_len
        advantages = torch.cat(all_adv)
        values_targ = torch.cat(all_vtarg)

        # Normalize advantages
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalizze values if using normalized_rmse
        if value_loss == 'normalized_rmse':
            # Update value normalization statistics (EMA)
            batch_mean = values_targ.mean().item()
            batch_std = values_targ.std().item() + 1e-8
            self.value_mean = ema_alpha * self.value_mean + (1 - ema_alpha) * batch_mean
            self.value_std = ema_alpha * self.value_std + (1 - ema_alpha) * batch_std
            values_targ = (values_targ - self.value_mean) / self.value_std

        # Do PPO epochs
        for ppo_ep in range(ppo_epochs):
            # Reset accumulators each PPO epoch
            sum_policy_loss = 0.0
            sum_value_loss = 0.0
            sum_entropy = 0.0
            sum_total_loss = 0.0
            num_updates = 0
            indices = torch.randperm(states.shape[0])
            for start in range(0, states.shape[0] - minibatch_size + 1, minibatch_size):
                mb = indices[start:start + minibatch_size]

                probs, values = self.network.forward(states[mb])
                distribution = torch.distributions.Categorical(probs)

                logp_cur = distribution.log_prob(actions[mb])
                ratio = torch.exp(logp_cur - logp_old[mb])

                surrogate1 = ratio * advantages[mb]
                surrogate2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages[mb]
                policy_loss = torch.min(surrogate1, surrogate2)
                if value_loss == 'mse':
                    v_loss = F.mse_loss(values, values_targ[mb])
                elif value_loss == 'rmse':
                    v_loss = ((values - values_targ[mb]) ** 2).mean().sqrt()
                elif value_loss == 'normalized_rmse':
                    v_loss = F.mse_loss(values, values_targ[mb])
                elif value_loss == 'relative_rmse':
                    v_loss = (((values - values_targ[mb]) / (values_targ[mb].abs() + 1e-3)) ** 2).mean()
                entropy = distribution.entropy().mean()

                loss = -(clip_coef * policy_loss.mean() - value_coef * v_loss + entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                sum_policy_loss += policy_loss.mean().item()
                sum_value_loss += v_loss.item()
                sum_entropy += entropy.item()
                sum_total_loss += loss.item()
                num_updates += 1

        # Return last epoch's stats
        return (sum_policy_loss / num_updates, sum_value_loss / num_updates,
                sum_entropy / num_updates, sum_total_loss / num_updates)


    # Helper functions
    def compute_advantage(self, rewards, values, gamma, lam):  # Eq 11 in PPO paper (GAE)
        t_max = rewards.shape[0]
        advantages = torch.zeros(t_max, dtype=torch.float32)
        last_gae = 0.0
        for t in reversed(range(t_max)):
            next_value = values[t + 1] if t < t_max - 1 else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        values_targ = advantages + values
        return advantages.detach(), values_targ.detach()




