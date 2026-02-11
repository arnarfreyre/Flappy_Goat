from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE


# Helper functions
def normalize_game_state(state):
    means = torch.tensor([256.0, 0.0, 200.0, 200.0, 200.0, 400.0, 200.0, 200.0])
    stds = torch.tensor([128.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    return (state - means) / stds

def state_to_tensor(state):
    tensor_state = torch.tensor(list(state.values()), dtype=torch.float32)
    tensor_state = normalize_game_state(tensor_state)

    return tensor_state

class FlappyNetwork(nn.Module):
    def __init__(self, hidden_layers=[64, 64], initial_weights=None):
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
        logits = self.actor(y)
        value = self.value(y).squeeze(-1)

        return logits, value

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
        logits, value = self.network.forward(state)
        # Convert to distribution
        distribution = torch.distributions.Categorical(logits=logits)
        # Get action
        if mode == 'Explore': # Sample from distribution
            action = distribution.sample()
        elif mode =='Exploit': # Choose most probable value
            action = torch.argmax(logits, dim=-1)
        else: # Something has gone wrong
            print('How did you manage to mess up the mode?? Defaulting to Explore I guess...')
            action = distribution.sample()
        # Get the log-probability
        logp = distribution.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def play_episode(self, episode, mode='Explore'):
        # Play an entire game
        episode.reset_game()
        game_path = []
        while not episode.game_over():
            # Get the current state
            state = state_to_tensor(episode.getGameState())
            # Decide on the action
            action, logp, value = self.get_action(state, mode)
            # Do the action and get reward
            reward = episode.act(episode.getActionSet()[action])
            # Store experience point
            game_path.append(torch.cat([
                state,
                torch.tensor([float(action), float(reward), float(logp), float(value)],
                             dtype=torch.float32)
                ]))
        # Store game experience
        self.memory = torch.stack(game_path)

    # Training functions
    def run_training(self, gamma, lam, clip_eps, value_coef, entropy_coef, max_grad_norm,
                     learning_rate, ppo_epochs, num_epochs, target_steps=800, minibatch_size=128, print_freq=100,
                     ema_alpha=0.9, value_loss='mse', reward_values=None):
        if reward_values is None:
            shift = 5
        else:
            shift = - reward_values['loss']
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        game = FlappyBird()
        ple_kwargs = dict(fps=30, display_screen=False, force_fps=True)
        if reward_values is not None:
            ple_kwargs['reward_values'] = reward_values
        episode = PLE(game, **ple_kwargs)
        episode.init()
        total_rewards = []
        for epoch in range(num_epochs):
            # Anneal learning rate
            lr = learning_rate * (1.0 - epoch / num_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # Collect fresh games until target timesteps reached
            batch = []
            epoch_rewards = []
            total_steps = 0
            while total_steps < target_steps:
                self.play_episode(episode, mode='Explore')
                batch.append(self.memory)
                epoch_rewards.append(self.memory[:, 9].sum().item())
                total_steps += self.memory.shape[0]
            total_rewards.append(sum(epoch_rewards) / len(epoch_rewards) + shift)
            # Train on the full batch, then discard
            l_clip, l_vf, l_ent, loss = self.train(batch, gamma, lam, clip_eps, value_coef,
                                                    entropy_coef, max_grad_norm, ppo_epochs, minibatch_size,
                                                    ema_alpha, value_loss)

            if (epoch + 1) % print_freq == 0:
                recent = total_rewards[-print_freq:]
                avg_pipes = sum(recent) / len(recent)
                print(f"Epoch {epoch+1:5d} | "
                      f"Avg Pipes: {avg_pipes:7.2f} | "
                      f"L_clip: {l_clip:.4f} | "
                      f"L_vf: {l_vf:.4f} | "
                      f"L_ent: {l_ent:.4f} | "
                      f"Loss: {loss:.4f}")

    def train(self, batch, gamma, lam, clip_eps, value_coef, entropy_coef, max_grad_norm, ppo_epochs, minibatch_size,
              ema_alpha=0.9, value_loss='mse'):
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

        # Recompute V(s) with current network
        with torch.no_grad():
            _, fresh_values = self.network.forward(states)
        # Denormalize to raw scale for advantage computation
        if value_loss == 'normalized_rmse':
            fresh_values = fresh_values * self.value_std + self.value_mean

        # Compute advantages per-episode, then normalize across the batch
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if value_loss == 'normalized_rmse':
            # Update value normalization statistics (EMA)
            batch_mean = values_targ.mean().item()
            batch_std = values_targ.std().item() + 1e-8
            self.value_mean = ema_alpha * self.value_mean + (1 - ema_alpha) * batch_mean
            self.value_std = ema_alpha * self.value_std + (1 - ema_alpha) * batch_std
            values_targ = (values_targ - self.value_mean) / self.value_std

        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy = 0.0
        sum_total_loss = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            indices = torch.randperm(states.shape[0])
            for start in range(0, states.shape[0], minibatch_size):
                mb = indices[start:start + minibatch_size]

                logits, values = self.network.forward(states[mb])
                distribution = torch.distributions.Categorical(logits=logits)

                logp_cur = distribution.log_prob(actions[mb])
                ratio = torch.exp(logp_cur - logp_old[mb])

                surrogate1 = ratio * advantages[mb]
                surrogate2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages[mb]
                policy_loss = torch.min(surrogate1, surrogate2)
                if value_loss == 'mse':
                    v_loss = F.mse_loss(values, values_targ[mb])
                elif value_loss == 'normalized_rmse':
                    v_loss = F.mse_loss(values, values_targ[mb])
                elif value_loss == 'relative_rmse':
                    v_loss = (((values - values_targ[mb]) / (values_targ[mb].abs() + 1e-3)) ** 2).mean()
                entropy = distribution.entropy().mean()

                loss = -(policy_loss.mean() - value_coef * v_loss + entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                sum_policy_loss += policy_loss.mean().item()
                sum_value_loss += v_loss.item()
                sum_entropy += entropy.item()
                sum_total_loss += loss.item()
                num_updates += 1

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




