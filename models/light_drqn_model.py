import torch
import torch.nn as nn

class DRQN(nn.Module):
    """
    Optimized DRQN for CPU.
    - Uses a smaller CNN for 7x7 MiniGrid inputs.
    - Reduces hidden_size to lower the computational load on the CPU.
    """

    def __init__(self, n_actions, hidden_size=128, sequence_length=8):
        super(DRQN, self).__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # MiniGrid inputs are typically (3, 7, 7)
        # We use a compact CNN that a CPU can process instantly
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # For a 7x7 input, two 2x2 convs (stride 1) leave a 5x5 feature map.
        # 32 channels * 5 * 5 = 800
        self.conv_out_size = 800

        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x, hidden=None):
        # Handle input: (Batch, Seq, C, H, W) or (Batch, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len, channels, h, w = x.shape

        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0

        # Collapse batch and seq for the CNN
        x = x.reshape(batch_size * seq_len, channels, h, w)
        x = self.conv(x)
        
        # Reshape back for LSTM: (Batch, Seq, Features)
        x = x.view(batch_size, seq_len, self.conv_out_size)

        # LSTM step
        x, hidden = self.lstm(x, hidden)

        # Only take the Q-values for the final step in the sequence
        x = self.fc(x[:, -1, :])

        return x, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)