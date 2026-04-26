import random
import numpy as np
from collections import deque


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for DRQN.

    Unlike standard DQN which samples random transitions,
    DRQN needs to sample random SEQUENCES of transitions
    to train the LSTM hidden state properly.

    Stores complete episodes and samples fixed-length sequences from them.
    """

    def __init__(self, capacity=1000, sequence_length=8):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        self.success_buffer = deque(maxlen=capacity // 2)
        self.current_episode = []
        self.episode_reward = 0

    def push_transition(self, obs, action, reward, next_obs, done, terminal):
        """Add a single transition to the current episode."""
        self.current_episode.append((obs, action, reward, next_obs, done, terminal))
        self.episode_reward += reward

        # Episode is done — store it and start a new one
        if done:
            if len(self.current_episode) >= self.sequence_length:
                if self.episode_reward > 0:
                    self.success_buffer.append(list(self.current_episode))
                self.buffer.append(list(self.current_episode))
            self.current_episode = []
            self.episode_reward = 0

    def sample(self, batch_size):
        """
        Sample batch_size sequences of length sequence_length.
        Each sequence comes from a random point in a random episode.
        """
        sequences = []
        success_count = min(batch_size // 2, len(self.success_buffer))

        # Sample from success buffer
        while len(sequences) < success_count:
            if len(self.success_buffer) == 0:
                break
            episode = random.choice(self.success_buffer)
            if len(episode) < self.sequence_length:
                continue
            start = random.randint(0, len(episode) - self.sequence_length)
            seq = episode[start:start + self.sequence_length]
            sequences.append(seq)

        # Sample from regular buffer
        while len(sequences) < batch_size:
            # Pick a random episode
            episode = random.choice(self.buffer)

            # Episode must be long enough
            if len(episode) < self.sequence_length:
                continue

            # Pick a random starting point in the episode
            start = random.randint(0, len(episode) - self.sequence_length)
            seq = episode[start:start + self.sequence_length]
            sequences.append(seq)

        # Unzip into separate arrays
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_terminals = [], [], [], [], [], []

        for seq in sequences:
            obs, actions, rewards, next_obs, dones, terminals = zip(*seq)
            batch_obs.append(np.array(obs))
            batch_actions.append(np.array(actions))
            batch_rewards.append(np.array(rewards))
            batch_next_obs.append(np.array(next_obs))
            batch_dones.append(np.array(dones))
            batch_terminals.append(np.array(terminals))

        return (
            np.array(batch_obs),       # (batch, seq, C, H, W) — shape depends on wrapper
            np.array(batch_actions),   # (batch, seq)
            np.array(batch_rewards),   # (batch, seq)
            np.array(batch_next_obs),  # (batch, seq, C, H, W) — shape depends on wrapper
            np.array(batch_dones),     # (batch, seq)
            np.array(batch_terminals)
        )

    def __len__(self):
        return len(self.buffer)

    def ready(self, min_episodes=10):
        """Check if buffer has enough episodes to start training."""
        return len(self.buffer) >= min_episodes