import torch
import torch.nn as nn


class DRQN(nn.Module):
    """
    Optimized DRQN for CPU.
    - Uses symbolic MiniGrid observations instead of pixels
    - Smaller CNN for faster CPU training
    - LSTM for explicit memory across timesteps
    """
    def __init__(self, n_actions, hidden_size=128, sequence_length=8, obs_shape=(3, 7, 7)):
        super(DRQN, self).__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        c, h, w = obs_shape

        # Compact CNN for small symbolic grid
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size dynamically
        dummy = torch.zeros(1, c, h, w)
        self.conv_out_size = self.conv(dummy).shape[1]
        print(f"Conv output size: {self.conv_out_size}")

        # LSTM — memory mechanism
        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Q-value head
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq, C, H, W) or (batch, C, H, W)
        hidden: LSTM hidden state
        """
        # Add sequence dimension if single step
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len, c, h, w = x.shape

        # Normalize
        x = x.float() / 255.0

        # Run CNN on each frame
        x = x.reshape(batch_size * seq_len, c, h, w)
        x = self.conv(x)

        # Reshape for LSTM
        x = x.view(batch_size, seq_len, self.conv_out_size)

        # LSTM — carries memory across timesteps
        x, hidden = self.lstm(x, hidden)

        # Q-values from last timestep only
        x = self.fc(x[:, -1, :])

        return x, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """Reset hidden state at episode start."""
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)