import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from wrappers.atari_wrapper import make_atari_env
from models.dqn_model import NatureDQN
from torch.utils.tensorboard import SummaryWriter
import os

CHECKPOINT_PATH = "checkpoints/dqn_checkpoint.pth"

def save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "step": step,
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode_rewards": episode_rewards,
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(policy_net, target_net, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, []
    ckpt = torch.load(CHECKPOINT_PATH)
    policy_net.load_state_dict(ckpt["policy_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']}")
    return ckpt["step"], ckpt["episode_rewards"]

def train():
    # Hyperparameters
    ENV_NAME = "ALE/Breakout-v5"
    LR = 1e-4
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 20000
    LEARNING_STARTS = 10000
    TARGET_UPDATE_FREQ = 1000
    TOTAL_STEPS = 500000
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 500000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    env = make_atari_env(ENV_NAME, render_mode=None)

    policy_net = NatureDQN(env.action_space.n).to(device)
    target_net = NatureDQN(env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=REPLAY_SIZE)

    # Load checkpoint if it exists
    start_step, episode_rewards = load_checkpoint(policy_net, target_net, optimizer)

    writer = SummaryWriter(log_dir="logs/dqn_breakout_v1", purge_step=start_step)

    obs, _ = env.reset()
    episode_reward = 0

    for step in range(start_step, TOTAL_STEPS):
        # Choose Action (Epsilon-Greedy Policy)
        epsilon = max(EPS_END, EPS_START - step / EPS_DECAY)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.tensor(np.array(obs), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_t).argmax().item()

        # Step in Environment
        next_obs, reward, done, truncated, _ = env.step(action)
        memory.append((obs, action, reward, next_obs, done))
        obs = next_obs
        episode_reward += reward

        # Optimize the loss function
        if len(memory) > LEARNING_STARTS:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Cast to float32 and normalize pixels
            states = torch.tensor(np.array(states), dtype=torch.uint8).to(device)
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.uint8).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            # Current Q values
            current_q = policy_net(states).gather(1, actions).squeeze()

            # Target Q values (Bellman Equation)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

            loss = nn.SmoothL1Loss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Losses/TD_Loss", loss.item(), step)

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Logging & Periodic Saving
        if done or truncated:
            print(f"Step: {step} | Reward: {episode_reward} | Epsilon: {epsilon:.2f}")

            episode_rewards.append(episode_reward)
            writer.add_scalar("Charts/Episode_Reward", episode_reward, step)
            writer.add_scalar("Charts/Epsilon", epsilon, step)
            if len(episode_rewards) >= 100:
                mean_100 = np.mean(episode_rewards[-100:])
                writer.add_scalar("Charts/Mean100_Reward", mean_100, step)

            # Save checkpoint every 5000 steps
            if step > 0 and step % 5000 < 100:
                save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards)
                torch.save(policy_net.state_dict(), "dqn_breakout_checkpoint.pth")

            obs, _ = env.reset()
            episode_reward = 0

    writer.close()

if __name__ == "__main__":
    train()