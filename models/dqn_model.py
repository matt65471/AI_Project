import torch
import torch.nn as nn

class NatureDQN(nn.Module):
    def __init__(self, n_actions):
        super(NatureDQN, self).__init__()
        
        # input shape: (4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened features after the conv layers
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Normalize pixel values from [0, 255] to [0, 1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)