# dqn_agent.py

import os
import random
from collections import deque
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim

    def add(self, state, action, reward, next_state, done):
        # store flattened numpy arrays for states
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.stack(states, axis=0)
        next_states = np.stack(next_states, axis=0)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.95,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: str = "cpu",
        # for evaluation: 252 for daily, 52 for weekly, etc.
        annualization_factor: float = 252.0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.annualization_factor = annualization_factor

        self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim)

        self.total_steps = 0  # for epsilon decay & target updates

    def _select_action(self, state_vec: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection given a single flattened state vector.
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def _optimize_model(self):
        """
        Sample a batch from replay buffer and perform one gradient step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Q(s, a)
        q_values = self.q_net(states)  # (batch, n_actions)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def _linear_epsilon_decay(
        self,
        step: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
    ) -> float:
        if step >= epsilon_decay_steps:
            return epsilon_end
        frac = step / float(epsilon_decay_steps)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    def train(
        self,
        env,
        n_episodes: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        max_steps_per_episode: Optional[int] = None,
        # NEW: optional checkpointing (e.g., every 10 episodes)
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> List[float]:
        """
        Train DQN on the given environment.

        Returns:
            List of total rewards per episode.
        """
        episode_rewards: List[float] = []

        self.total_steps = 0  # reset counter if you want fresh schedule

        for episode in range(n_episodes):
            obs, info = env.reset()
            state = np.asarray(obs, dtype=np.float32).reshape(-1)

            done = False
            truncated = False
            total_reward = 0.0
            steps_in_episode = 0

            while not (done or truncated):
                epsilon = self._linear_epsilon_decay(
                    self.total_steps, epsilon_start, epsilon_end, epsilon_decay_steps
                )

                action = self._select_action(state, epsilon)

                next_obs, reward, done, truncated, info = env.step(action)
                next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)

                self.replay_buffer.add(
                    state, action, reward, next_state, done or truncated
                )

                state = next_state
                total_reward += reward
                steps_in_episode += 1
                self.total_steps += 1

                # optimize
                self._optimize_model()

                # target network update
                if self.total_steps % self.target_update_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                if (
                    max_steps_per_episode is not None
                    and steps_in_episode >= max_steps_per_episode
                ):
                    break

            episode_rewards.append(total_reward)

            # simple logging
            if (episode + 1) % 10 == 0:
                recent = episode_rewards[-10:]
                avg_recent = sum(recent) / len(recent)
                print(
                    f"Episode {episode + 1}/{n_episodes}, "
                    f"Reward: {total_reward:.4f}, "
                    f"Avg(10): {avg_recent:.4f}, "
                    f"Epsilon: {epsilon:.3f}"
                )

            # optional checkpointing
            if (
                checkpoint_dir is not None
                and checkpoint_interval is not None
                and (episode + 1) % checkpoint_interval == 0
            ):
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"dqn_ep{episode + 1}.pt"
                )
                torch.save(
                    {
                        "episode": episode + 1,
                        "model_state_dict": self.q_net.state_dict(),
                        "target_model_state_dict": self.target_q_net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "total_steps": self.total_steps,
                        "episode_rewards": episode_rewards,
                        "obs_dim": self.obs_dim,
                        "n_actions": self.n_actions,
                        "gamma": self.gamma,
                    },
                    ckpt_path,
                )
                print(f"[Checkpoint] Saved to {ckpt_path}")

        return episode_rewards

    def evaluate(
        self,
        env,
        annualization_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the greedy DQN policy (epsilon=0) on the given environment.

        Args:
            env: environment
            annualization_factor: if provided, overrides the agent's default.
                                  Use 252 for daily, 52 for weekly, etc.

        Returns:
            dict with keys:
                - total_reward
                - mean_reward
                - rewards (list)
                - positions (list of -1, 0, 1)
                - sharpe (annualized)
        """
        obs, info = env.reset()
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        done = False
        truncated = False

        rewards: List[float] = []
        positions: List[int] = []

        while not (done or truncated):
            # epsilon = 0 => purely greedy
            action = self._select_action(state, epsilon=0.0)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)

            rewards.append(float(reward))
            positions.append(int(action - 1))  # map 0/1/2 -> -1/0/1

            state = next_state

        rewards_arr = np.array(rewards, dtype=np.float32)
        total_reward = float(rewards_arr.sum())
        mean_reward = float(rewards_arr.mean()) if len(rewards_arr) > 0 else 0.0
        std_reward = float(rewards_arr.std()) if len(rewards_arr) > 0 else 0.0

        af = annualization_factor if annualization_factor is not None else self.annualization_factor

        sharpe = (
            mean_reward / (std_reward + 1e-8) * np.sqrt(af)
            if len(rewards_arr) > 1
            else 0.0
        )

        return {
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "rewards": rewards,
            "positions": positions,
            "sharpe": sharpe,
        }
