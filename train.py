"""Training script for DQN on either Pong (Atari) or MiniGrid Memory.

Usage
-----
  python train.py                          # default settings
  python train.py --total-steps 1000000
  python train.py --domain atari --env ALE/Pong-v5
  python train.py --domain minigrid --env MiniGrid-MemoryS9-v0

The script logs to the terminal and saves:
  - checkpoints/dqn_<domain>_step_<N>.pt  every --save-freq steps
  - logs/training_log.csv             episode-level CSV log
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from atari_preprocessing import make_dqn_env
from dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Atari Pong or MiniGrid Memory.")
    p.add_argument("--domain",       choices=["atari", "minigrid"], default="minigrid")
    p.add_argument("--env",          default=None,
                   help="Env id; defaults to ALE/Pong-v5 for atari, MiniGrid-MemoryS9-v0 for minigrid")
    p.add_argument("--total-steps",  type=int,   default=1_000_000)
    p.add_argument("--replay-capacity", type=int, default=250_000)
    p.add_argument("--replay-start",    type=int, default=20_000,
                   help="Steps of random exploration before learning begins")
    p.add_argument("--target-update",   type=int, default=10_000,
                   help="Frequency (in gradient updates) to sync target network")
    p.add_argument("--batch-size",      type=int, default=32)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--lr",              type=float, default=0.00025)
    p.add_argument("--eps-start",       type=float, default=1.0)
    p.add_argument("--eps-end",         type=float, default=0.05)
    p.add_argument("--eps-anneal",      type=int,   default=500_000)
    p.add_argument("--save-freq",       type=int,   default=500_000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--eval-freq",       type=int,   default=100_000,
                   help="Run evaluation every N steps")
    p.add_argument("--eval-episodes",   type=int,   default=10)
    p.add_argument("--log-freq",        type=int,   default=10,
                   help="Print summary every N episodes")
    p.add_argument("--resume",          default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--log-dir",         default="logs")
    p.add_argument("--device",          default=None,
                   help="cuda / mps / cpu  (auto-detected if omitted)")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent: DQNAgent, domain: str, env_id: str, n_episodes: int) -> float:
    """Run n_episodes with ε=0.05 and return mean total reward."""
    eval_env = make_dqn_env(
        domain,
        env_id,
        clip_reward=False,
    )

    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            # Use a fixed low ε for evaluation (common practice).
            import random as _random
            if _random.random() < 0.05:
                action = eval_env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
                    action = int(agent.online_net(state).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    eval_env.close()
    return float(np.mean(total_rewards))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir        = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_id = args.env or ("ALE/Pong-v5" if args.domain == "atari" else "MiniGrid-MemoryS9-v0")

    # Environment
    env = make_dqn_env(args.domain, env_id)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    print(f"Domain      : {args.domain}")
    print(f"Environment : {env_id}")
    print(f"Actions     : {n_actions}")
    print(f"Action IDs  : 0..{n_actions - 1}")
    print(f"Obs shape   : {obs_shape}")

    # Agent
    agent = DQNAgent(
        n_actions=n_actions,
        obs_shape=obs_shape,
        device=args.device,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        replay_start_size=args.replay_start,
        target_update_freq=args.target_update,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_anneal_steps=args.eps_anneal,
        lr=args.lr,
    )
    print(f"Device      : {agent.device}")

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}  (step {agent.steps_done:,})")

    # CSV logger
    log_path = log_dir / "training_log.csv"
    csv_file  = open(log_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if log_path.stat().st_size == 0:
        csv_writer.writerow(
            ["step", "episode", "ep_reward", "ep_length", "epsilon",
             "loss", "eval_mean_reward", "elapsed_s"]
        )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    obs, _ = env.reset()
    ep_reward  = 0.0
    ep_length  = 0
    ep_losses: list[float] = []
    episode    = 0
    eval_score = float("nan")

    # Recent episode rewards for the rolling average displayed in the progress bar.
    recent_rewards: list[float] = []

    t0 = time.time()
    pbar = tqdm(total=args.total_steps, initial=agent.steps_done,
                unit="step", dynamic_ncols=True)

    while agent.steps_done < args.total_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, float(reward), next_obs, done)
        loss = agent.update()

        if loss is not None:
            ep_losses.append(loss)

        ep_reward += float(reward)
        ep_length += 1
        obs = next_obs

        pbar.update(1)

        if done:
            episode += 1
            recent_rewards.append(ep_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            eps       = agent.current_epsilon()

            if episode % args.log_freq == 0:
                mean_r = np.mean(recent_rewards)
                pbar.set_postfix(
                    ep=episode,
                    r=f"{ep_reward:.1f}",
                    r100=f"{mean_r:.2f}",
                    eps=f"{eps:.3f}",
                    loss=f"{mean_loss:.4f}" if ep_losses else "N/A",
                )

            csv_writer.writerow([
                agent.steps_done, episode, ep_reward, ep_length,
                f"{eps:.4f}", f"{mean_loss:.6f}", eval_score,
                f"{time.time() - t0:.1f}",
            ])
            csv_file.flush()

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            ep_losses = []

        # Periodic evaluation
        if args.eval_freq > 0 and agent.steps_done % args.eval_freq == 0:
            eval_score = evaluate(agent, args.domain, env_id, args.eval_episodes)
            tqdm.write(
                f"[Eval @ {agent.steps_done:,}]  mean reward = {eval_score:.2f}"
                f"  (over {args.eval_episodes} episodes)"
            )

        # Periodic checkpoint
        if agent.steps_done % args.save_freq == 0:
            ckpt_path = checkpoint_dir / f"dqn_{args.domain}_step_{agent.steps_done}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[Saved] {ckpt_path}")

    pbar.close()
    env.close()
    csv_file.close()

    # Final checkpoint
    final_path = checkpoint_dir / f"dqn_{args.domain}_final.pt"
    agent.save(final_path)
    print(f"\nTraining complete.  Final checkpoint saved to {final_path}")
    print(f"Total time: {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    train(parse_args())
