import torch
import torch.nn as nn


class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) — adds LSTM after conv stack.
    This gives the agent explicit memory across timesteps,
    which is required for MiniGrid MemoryEnv where the agent
    must remember an object seen early in the episode.

    Architecture:
        Conv stack (same as NatureDQN) → LSTM → Linear Q-head
    """

    def __init__(self, n_actions, hidden_size=512, sequence_length=8):
        super(DRQN, self).__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Same conv stack as NatureDQN — input (B, 4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Conv output is 64 * 7 * 7 = 3136
        self.conv_out_size = 3136

        # LSTM replaces the FC layer — this is the memory mechanism
        # Takes conv features and maintains hidden state across timesteps
        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            batch_first=True  # (batch, seq, features)
        )

        # Q-value output head
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, 4, 84, 84) for sequence input
           or (batch, 4, 84, 84) for single step
        hidden: (h, c) LSTM hidden state — None resets to zeros
        """
        # Handle single step input — add sequence dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 4, 84, 84)

        batch_size, seq_len = x.shape[0], x.shape[1]

        # Normalize pixels
        x = x.float() / 255.0

        # Run conv on each frame in the sequence
        x = x.view(batch_size * seq_len, 4, 84, 84)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, self.conv_out_size)

        # Run LSTM — hidden state carries memory across timesteps
        x, hidden = self.lstm(x, hidden)

        # Q-values from last timestep
        x = self.fc(x[:, -1, :])

        return x, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden state to zeros at episode start."""
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)