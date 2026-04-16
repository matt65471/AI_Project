"""Three-layer convolutional Q-head for DQN-style Atari inputs (4×84×84)."""

from __future__ import annotations

import torch
from torch import nn


# Spatial: 84 → 20 → 9 → 7 with kernel/stride as in Mnih et al. conv stack.
_CONV_FLAT_DIM = 64 * 7 * 7


class DQNConv(nn.Module):
    """
    Conv stack: 32×8×8/s4 → 64×4×4/s2 → 64×3×3/s1, each followed by ReLU,
    then flatten and a linear layer to one Q-value per action.

    Expects ``forward`` input shaped ``(B, in_channels, 84, 84)`` (e.g. from
    :class:`~atari_preprocessing.DQNAtariPreprocessWrapper` after ``float``
    and optional ``/ 255.0``).

    **Action count:** set ``n_actions`` to ``env.action_space.n`` (or the
    size of your reduced action set). For example, ``ALE/Pong-v5`` has six
    discrete actions; if you only need three logits, use a three-action env
    or an action wrapper and match ``n_actions`` to that MDP.
    """

    def __init__(self, *, in_channels: int = 4, n_actions: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Linear(_CONV_FLAT_DIM, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


if __name__ == "__main__":
    for n_act in (3, 6):
        net = DQNConv(n_actions=n_act)
        batch = torch.randn(2, 4, 84, 84)
        out = net(batch)
        assert out.shape == (2, n_act), out.shape
    print("DQNConv shape checks OK.")
