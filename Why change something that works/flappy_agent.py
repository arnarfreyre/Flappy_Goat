from math import inf
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd

from ple.games.flappybird import FlappyBird
from ple import PLE


# Helper functions
def normalize_game_state(state):
    means = torch.tensor([150.0, 0.0, 76.0, 108.0, 208.0, 226.0, 108.0, 208.0])
    stds = torch.tensor([44.0, 5.0, 44.0, 48.0, 48.0, 44.0, 48.0, 48.0])
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
        probs = F.softmax(self.actor(y), dim=-1)
        value = self.value(y).squeeze(-1)

        return probs, value

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
        #observed_states = {'player_pos':[], 'player_vel':[], 'next_pipe_dist':[], 'next_pipe_top':[],
        #                                        'next_pipe_bot':[], 'next_next_pipe_dist':[], 'next_next_pipe_top':[],
        #                                        'next_next_pipe_bot':[]}
        # Play an entire game
        episode.reset_game()
        game_path = []
        nr_pipes = 0
        while not episode.game_over():
            # Get the current state
            state = episode.getGameState()
            #observed_states['player_pos'].append(state['player_y'])
            #observed_states['player_vel'].append(state['player_vel'])
            #observed_states['next_pipe_dist'].append(state['next_pipe_dist_to_player'])
            #observed_states['next_pipe_top'].append(state['next_pipe_top_y'])
            #observed_states['next_pipe_bot'].append(state['next_pipe_bottom_y'])
            #observed_states['next_next_pipe_dist'].append(state['next_next_pipe_dist_to_player'])
            #observed_states['next_next_pipe_top'].append(state['next_next_pipe_top_y'])
            #observed_states['next_next_pipe_bot'].append(state['next_next_pipe_bottom_y'])

            state = state_to_tensor(state)
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
            if reward > 0:
                nr_pipes += int(reward)
                if print_freq is not None and nr_pipes % print_freq == 0:
                    print()
                    print(nr_pipes)
                if max_pipes is not None and nr_pipes >= max_pipes:
                    break
        # Store game experience
        self.memory = torch.stack(game_path)
        # return observed_states

    # Training functions
    def run_training(self, gamma, lam, clip_eps, clip_coef, value_coef, entropy_coef, max_grad_norm,
                     learning_rate, ppo_epochs, num_epochs, target_steps=800, minibatch_size=128, print_freq=100,
                     ema_alpha=0.9, value_loss='mse', reward_values=None, normalize_advantage=True):
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
        total_l_clips = []
        total_l_vfs = []
        total_l_ents = []
        total_losses = []
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
            # Enable debug diagnostics on print_freq epochs
            self._debug = ((epoch + 1) % print_freq == 0)
            l_clip, l_vf, l_ent, loss = self.train(batch, gamma, lam, clip_eps, clip_coef, value_coef,
                                                    entropy_coef, max_grad_norm, ppo_epochs, minibatch_size,
                                                    ema_alpha, value_loss, normalize_advantage)
            total_l_clips.append(l_clip)
            total_l_vfs.append(l_vf)
            total_l_ents.append(l_ent)
            total_losses.append(loss)

            if (epoch + 1) % print_freq == 0:
                recent = total_rewards[-print_freq:]
                avg_pipes = sum(recent) / len(recent)
                recent_l_clips = total_l_clips[-print_freq:]
                recent_l_vfs = total_l_vfs[-print_freq:]
                recent_l_ents = total_l_ents[-print_freq:]
                recent_losses = total_losses[-print_freq:]
                print(f"Epoch {epoch+1:5d} | "
                      f"Avg Pipes: {avg_pipes:7.2f} | "
                      f"L_clip: {sum(recent_l_clips) / len(recent_l_clips):.4f} | "
                      f"L_vf: {sum(recent_l_vfs) / len(recent_l_vfs):.4f} | "
                      f"L_ent: {sum(recent_l_ents) / len(recent_l_ents):.4f} | "
                      f"Loss: {sum(recent_losses) / len(recent_losses):.4f}")

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

        # Use stored V(s) from rollout (column 11) — identical to recomputing
        # since the network hasn't been updated yet at this point
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
        # Normalize advantages (zero mean, unit variance)
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # DEBUG: advantage and value diagnostics
        if hasattr(self, '_debug') and self._debug:
            print(f"  [DEBUG] advantages  | mean: {advantages.mean():.4f} | std: {advantages.std():.4f} | min: {advantages.min():.4f} | max: {advantages.max():.4f} | frac>0: {(advantages > 0).float().mean():.2f}", file=sys.stderr)
            print(f"  [DEBUG] values_targ | mean: {values_targ.mean():.4f} | std: {values_targ.std():.4f} | min: {values_targ.min():.4f} | max: {values_targ.max():.4f}", file=sys.stderr)
            print(f"  [DEBUG] fresh_vals  | mean: {fresh_values.mean():.4f} | std: {fresh_values.std():.4f} | min: {fresh_values.min():.4f} | max: {fresh_values.max():.4f}", file=sys.stderr)
            print(f"  [DEBUG] logp_old    | mean: {logp_old.mean():.4f} | std: {logp_old.std():.4f} | min: {logp_old.min():.4f} | max: {logp_old.max():.4f}", file=sys.stderr)
            print(f"  [DEBUG] batch: {len(batch)} episodes | {states.shape[0]} total steps", file=sys.stderr)
        # END DEBUG
        if value_loss == 'normalized_rmse':
            # Update value normalization statistics (EMA)
            batch_mean = values_targ.mean().item()
            batch_std = values_targ.std().item() + 1e-8
            self.value_mean = ema_alpha * self.value_mean + (1 - ema_alpha) * batch_mean
            self.value_std = ema_alpha * self.value_std + (1 - ema_alpha) * batch_std
            values_targ = (values_targ - self.value_mean) / self.value_std

        for ppo_ep in range(ppo_epochs):
            # Reset accumulators each PPO epoch — only the last epoch's values are returned
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

            if hasattr(self, '_debug') and self._debug:
                print(f"  [DEBUG] ppo_ep={ppo_ep} | avg L_clip: {sum_policy_loss / num_updates:.4f} | avg L_vf: {sum_value_loss / num_updates:.4f} | avg L_ent: {sum_entropy / num_updates:.4f}", file=sys.stderr)

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




