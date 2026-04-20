"""Evaluate a saved Nature DQN checkpoint on an Atari environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import ale_py  # noqa: F401 - registers ALE/* envs
import numpy as np
import torch

from atari_preprocessing import make_atari_dqn_env
from dqn_cnn import NatureDQN


def maybe_fire_to_start(env, obs):
    """Atari games like Pong need FIRE once after reset to begin play."""
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "get_action_meanings"):
        return obs, 0.0, False

    action_meanings = unwrapped.get_action_meanings()
    if "FIRE" not in action_meanings:
        return obs, 0.0, False

    fire_action = action_meanings.index("FIRE")
    obs, reward, terminated, truncated, _ = env.step(fire_action)
    return obs, float(reward), bool(terminated or truncated)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint")
    p.add_argument("--checkpoint", type=str, default="dqn_checkpoint.pt")
    p.add_argument("--env-id", type=str, default=None, help="Override env id from checkpoint args if provided.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps-per-episode", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render", action="store_true", help="Show live gameplay (uses render_mode='human').")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "online" not in ckpt:
        raise ValueError(f"Checkpoint at {path} does not contain expected 'online' weights.")
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    return ckpt["online"], saved_args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    online_state, saved_args = load_checkpoint(ckpt_path, device)
    env_id = args.env_id or saved_args.get("env_id", "ALE/Pong-v5")
    render_mode = "human" if args.render else None

    env = make_atari_dqn_env(env_id, render_mode=render_mode, frame_stack=4, screen_size=(84, 84))
    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)
    model = NatureDQN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    model.load_state_dict(online_state)
    model.eval()

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        ep_len = 0
        obs, fire_reward, fire_done = maybe_fire_to_start(env, obs)
        ep_return += fire_reward
        if fire_done:
            returns.append(ep_return)
            lengths.append(ep_len)
            print(f"Episode {ep + 1}/{args.episodes}: return={ep_return:.2f}, length={ep_len}")
            continue

        for _ in range(args.max_steps_per_episode):
            with torch.no_grad():
                s = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)
                action = int(model(s).argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if terminated or truncated:
                break

        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"Episode {ep + 1}/{args.episodes}: return={ep_return:.2f}, length={ep_len}")

    env.close()

    arr = np.asarray(returns, dtype=np.float32)
    print(
        f"\nAverage return over {args.episodes} episodes: "
        f"{arr.mean():.2f} +/- {arr.std(ddof=0):.2f} | min={arr.min():.2f}, max={arr.max():.2f}"
    )
    print(f"Average episode length: {float(np.mean(lengths)):.1f}")


if __name__ == "__main__":
    main()
