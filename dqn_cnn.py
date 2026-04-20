"""DQN-style conv Q-network (4×84×84) plus TD MSE loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
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


def dqn_td_loss(
    q_sa: torch.Tensor,
    rewards: torch.Tensor,
    max_q_next_target: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """
    Mean squared TD error matching Mnih et al.:

        L_i(θ_i) = E_{s,a,r,s'}[(y - Q(s, a; θ_i))²],   y = r + γ max_{a'} Q(s', a'; θ_i^-)

    ``dones`` should be 1 where the transition is terminal (no bootstrap), 0 otherwise.
    ``max_q_next_target`` is max_a' Q(s', a'; θ^-) already detached from the graph if needed.
    """
    nonterminal = 1.0 - dones.to(dtype=q_sa.dtype, device=q_sa.device)
    targets = rewards + gamma * nonterminal * max_q_next_target
    return F.mse_loss(q_sa, targets)


def compute_dqn_loss(
    online: nn.Module,
    target: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """
    Full forward for one batch: Q(s,a; θ), bootstrap target with θ^-, then MSE loss.

    Gradients flow into ``online`` only; ``target`` is evaluated under ``no_grad``.
    Autograd on the returned scalar gives
    E[(r + γ max Q(s',·; θ^-) - Q(s,a; θ)) ∇_θ Q(s,a; θ)] for sampled transitions.
    """
    q_all = online(states)
    actions_i = actions.view(-1, 1).long().to(device=q_all.device)
    q_sa = q_all.gather(1, actions_i).squeeze(1)
    rewards = rewards.to(device=q_sa.device, dtype=q_sa.dtype)
    dones = dones.to(device=q_sa.device, dtype=q_sa.dtype)

    with torch.no_grad():
        q_next = target(next_states)
        max_next = q_next.max(dim=1).values

    return dqn_td_loss(q_sa, rewards, max_next, dones, gamma=gamma)


if __name__ == "__main__":
    for n_act in (3, 6):
        net = DQNConv(n_actions=n_act)
        batch = torch.randn(2, 4, 84, 84)
        out = net(batch)
        assert out.shape == (2, n_act), out.shape
    print("DQNConv shape checks OK.")

    B, n_act = 8, 6
    online = DQNConv(n_actions=n_act)
    target = DQNConv(n_actions=n_act)
    target.load_state_dict(online.state_dict())

    states = torch.randn(B, 4, 84, 84)
    next_states = torch.randn(B, 4, 84, 84)
    actions = torch.randint(0, n_act, (B,))
    rewards = torch.randn(B)
    dones = torch.zeros(B)

    loss = compute_dqn_loss(
        online, target, states, actions, rewards, next_states, dones, gamma=0.99
    )
    loss.backward()
    assert any(p.grad is not None for p in online.parameters())
    assert all(p.grad is None for p in target.parameters())
    print("dqn_loss OK:", float(loss))
