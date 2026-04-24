"""Evaluation & visualisation for a trained DQN Atari/MiniGrid agent.

Usage
-----
  # Watch the agent play (renders to screen):
  python evaluate.py --domain atari --env ALE/Pong-v5 --checkpoint checkpoints/dqn_atari_final.pt --render

  # Silent evaluation over N episodes:
  python evaluate.py --domain minigrid --env MiniGrid-MemoryS9-v0 --checkpoint checkpoints/dqn_minigrid_final.pt --episodes 30

  # Plot training curves from the CSV log:
  python evaluate.py --plot logs/training_log.csv
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from atari_preprocessing import make_dqn_env
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DQN Atari/MiniGrid agent.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint file.")
    p.add_argument("--domain",    choices=["atari", "minigrid"], default="minigrid")
    p.add_argument("--agent",     choices=["dqn", "drqn"], default="dqn",
                   help="Feed-forward DQN or recurrent DQN (LSTM).")
    p.add_argument("--lstm-hidden", type=int, default=128,
                   help="Hidden size of the LSTM (DRQN only).")
    p.add_argument("--env",       default=None,
                   help="Env id; defaults to ALE/Pong-v5 for atari, MiniGrid-MemoryS9-v0 for minigrid")
    p.add_argument("--episodes",  type=int, default=10,
                   help="Number of evaluation episodes.")
    p.add_argument("--epsilon",   type=float, default=0.05,
                   help="Exploration rate used during evaluation.")
    p.add_argument("--render",    action="store_true",
                   help="Render the game to screen (human mode where supported).")
    p.add_argument("--plot",      default=None,
                   help="Path to training_log.csv; plot reward curves instead of playing.")
    p.add_argument("--device",    default=None)
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_episodes(
    agent,
    domain: str,
    env_id: str,
    n_episodes: int,
    epsilon: float,
    render: bool,
) -> list[float]:
    """Play n_episodes and return raw (unclipped) episode rewards."""
    render_mode = "human" if render else None
    env = make_dqn_env(
        domain,
        env_id,
        render_mode=render_mode,
        clip_reward=False,
    )

    is_recurrent = isinstance(agent, DRQNAgent)

    rewards = []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        agent.reset_hidden()
        ep_reward = 0.0
        done = False
        step = 0
        while not done:
            if is_recurrent:
                with torch.no_grad():
                    state = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(agent.device)
                    q, agent._hidden = agent.online_net(state, agent._hidden)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(q.reshape(-1).argmax().item())
            else:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
                        action = int(agent.online_net(state).argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
            step += 1

        rewards.append(ep_reward)
        print(f"  Episode {ep:3d}: reward = {ep_reward:+.0f}  ({step} steps)")

    env.close()
    return rewards


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(csv_path: str) -> None:
    """Plot episode rewards and optional eval scores from a CSV log."""
    import csv
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    steps, ep_rewards, eval_scores, eval_steps = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            ep_rewards.append(float(row["ep_reward"]))
            if row["eval_mean_reward"] not in ("nan", ""):
                eval_scores.append(float(row["eval_mean_reward"]))
                eval_steps.append(int(row["step"]))

    steps      = np.array(steps)
    ep_rewards = np.array(ep_rewards)

    # Smooth episode rewards with a rolling mean (window = 100)
    window = 100
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    smoothed_steps = steps[window - 1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ep_rewards, alpha=0.2, color="steelblue", linewidth=0.5,
            label="Episode reward")
    ax.plot(smoothed_steps, smoothed, color="steelblue", linewidth=2,
            label=f"Rolling mean (n={window})")
    if eval_scores:
        ax.scatter(eval_steps, eval_scores, color="darkorange", zorder=5,
                   label="Eval mean reward")
        ax.plot(eval_steps, eval_scores, color="darkorange", linewidth=1.5,
                linestyle="--")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("DQN training curve")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = csv_path.replace(".csv", "_curve.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.plot:
        plot_training(args.plot)
        return

    if args.checkpoint is None:
        raise ValueError("Provide --checkpoint or --plot.")

    env_id = args.env or ("ALE/Pong-v5" if args.domain == "atari" else "MiniGrid-MemoryS9-v0")

    # Determine n_actions from the env.
    tmp_env = make_dqn_env(args.domain, env_id)
    n_actions = tmp_env.action_space.n
    obs_shape = tmp_env.observation_space.shape
    tmp_env.close()

    if args.agent == "drqn":
        agent = DRQNAgent(
            n_actions=n_actions,
            obs_shape=obs_shape,
            device=args.device,
            lstm_hidden_size=args.lstm_hidden,
        )
    else:
        agent = DQNAgent(n_actions=n_actions, obs_shape=obs_shape, device=args.device)
    agent.load(args.checkpoint)
    agent.online_net.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Agent            : {args.agent}")
    print(f"Domain           : {args.domain}")
    print(f"Environment      : {env_id}")
    print(f"Steps trained    : {agent.steps_done:,}")
    print(f"Device           : {agent.device}")
    print(f"Running {args.episodes} episode(s) with ε={args.epsilon:.2f} …\n")

    rewards = run_episodes(
        agent, args.domain, env_id, args.episodes, args.epsilon, args.render
    )

    print(f"\nResults over {len(rewards)} episodes:")
    print(f"  Mean   : {np.mean(rewards):+.2f}")
    print(f"  Std    : {np.std(rewards):.2f}")
    print(f"  Min    : {np.min(rewards):+.0f}")
    print(f"  Max    : {np.max(rewards):+.0f}")


if __name__ == "__main__":
    main()
