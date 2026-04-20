"""Train Nature DQN (Mnih et al., 2015) on Atari (Gymnasium ALE)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import ale_py  # noqa: F401 — registers ALE/* envs
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from atari_preprocessing import make_atari_dqn_env
from dqn_cnn import NatureDQN, compute_dqn_loss
from dqn_replay import ReplayBuffer


def linear_epsilon(frame_idx: int, *, decay_frames: int = 1_000_000, eps_start: float = 1.0, eps_end: float = 0.1) -> float:
    """Linear anneal from ``eps_start`` to ``eps_end`` over ``decay_frames`` env steps (Nature)."""
    if frame_idx >= decay_frames:
        return eps_end
    return eps_start + (eps_end - eps_start) * (frame_idx / float(decay_frames))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nature DQN on Atari")
    p.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    p.add_argument("--total-steps", type=int, default=500_000, help="Environment steps (Nature uses tens of millions for full runs).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--replay-capacity", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=50_000)
    p.add_argument("--train-frequency", type=int, default=4, help="Env steps between each gradient update (Nature: 4).")
    p.add_argument("--target-update-freq", type=int, default=10_000, help="Sync target net every this many env steps (Nature C).")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--rmsprop-alpha", type=float, default=0.95)
    p.add_argument("--rmsprop-eps", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--epsilon-decay-frames", type=int, default=1_000_000)
    p.add_argument("--log-interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_atari_dqn_env(args.env_id, render_mode=None, frame_stack=4, screen_size=(84, 84))
    env.action_space.seed(args.seed)

    n_actions = int(env.action_space.n)
    obs_shape = env.observation_space.shape
    assert len(obs_shape) == 3

    online = NatureDQN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    target = NatureDQN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    target.load_state_dict(online.state_dict())

    optimizer = torch.optim.RMSprop(
        online.parameters(),
        lr=args.lr,
        alpha=args.rmsprop_alpha,
        eps=args.rmsprop_eps,
        momentum=0.0,
    )

    replay = ReplayBuffer(capacity=args.replay_capacity, obs_shape=tuple(obs_shape))

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    episodes = 0

    pbar = tqdm(range(args.total_steps), desc=args.env_id)
    for t in pbar:
        eps = linear_epsilon(t, decay_frames=args.epsilon_decay_frames)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)
                q = online(s)
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        r_clip = float(np.clip(reward, -1.0, 1.0))
        replay.push(obs, action, r_clip, next_obs, done)

        ep_return += float(reward)

        if done:
            episodes += 1
            pbar.set_postfix(ep_return=f"{ep_return:.1f}", episodes=episodes, eps=f"{eps:.3f}", refresh=False)
            obs, _ = env.reset()
            ep_return = 0.0
        else:
            obs = next_obs

        if t >= args.learning_starts and t % args.train_frequency == 0 and replay.size >= args.batch_size:
            states, actions_b, rewards_b, next_states, dones_b = replay.sample(args.batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_dqn_loss(
                online,
                target,
                states,
                actions_b,
                rewards_b,
                next_states,
                dones_b,
                gamma=args.gamma,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), args.grad_clip_norm)
            optimizer.step()

        if t > 0 and t % args.target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        if t > 0 and t % args.log_interval == 0 and t >= args.learning_starts:
            pbar.set_postfix(replay=replay.size, refresh=False)

    env.close()
    out = Path(__file__).resolve().parent / "dqn_checkpoint.pt"
    torch.save({"online": online.state_dict(), "args": vars(args)}, out)
    tqdm.write(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    main()
