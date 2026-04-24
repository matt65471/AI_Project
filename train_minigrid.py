import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from wrappers.minigrid_wrapper import make_minigrid_env
from dqn_model import NatureDQN
from torch.utils.tensorboard import SummaryWriter
import os

#torch.backends.nnpack.set_flags(False)  # Disable NNPACK to avoid potential issues on some platforms

def save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards, checkpoint_path):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "step": step,
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode_rewards": episode_rewards,
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(policy_net, target_net, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0, []
    ckpt = torch.load(checkpoint_path)
    policy_net.load_state_dict(ckpt["policy_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']}")
    return ckpt["step"], ckpt["episode_rewards"]

def train():
    # Hyperparameters
    ENV_NAME = "MiniGrid-MemoryS7-v0"
    SEED = 32

    # Same LR as paper
    LR = 0.00025

    # Discount factor — paper sets γ = 0.99 throughout
    GAMMA = 0.99

    # Minibatch size from paper
    BATCH_SIZE = 32

    # Reduced from paper's 1M
    REPLAY_SIZE = 100000

    # Start learning after 10k steps 
    LEARNING_STARTS = 10000

    # Target network update frequency — reduced from 10k for faster MiniGrid episodes
    TARGET_UPDATE_FREQ = 1000

    # MiniGrid needs fewer steps than Atari to show the failure mode
    TOTAL_STEPS = 500000

    EPS_START = 1.0
    EPS_END = 0.1

    # Anneal epsilon over first 250k steps
    EPS_DECAY_STEPS = 250000

    CHECKPOINT_PATH = f"checkpoints/dqn_minigrid_seed{SEED}_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Seed: {SEED}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Make environment
    env = make_minigrid_env(ENV_NAME)
    env.action_space.seed(SEED)

    policy_net = NatureDQN(env.action_space.n).to(device)
    target_net = NatureDQN(env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is never trained directly

    # RMSProp with same paper hyperparameters as Pong
    optimizer = optim.RMSprop(
        policy_net.parameters(),
        lr=LR,
        alpha=0.95,
        eps=0.01,
        momentum=0.95,
        centered=True
    )

    memory = deque(maxlen=REPLAY_SIZE)

    # Load checkpoint if it exists
    start_step, episode_rewards = load_checkpoint(policy_net, target_net, optimizer, CHECKPOINT_PATH)

    writer = SummaryWriter(log_dir=f"logs/dqn_minigrid_seed{SEED}", purge_step=start_step)

    obs, _ = env.reset()
    episode_reward = 0

    for step in range(start_step, TOTAL_STEPS):
        # Choose Action (Epsilon-Greedy Policy)
        # Anneal ε linearly from 1.0 to 0.1 over 250k steps
        epsilon = max(EPS_END, EPS_START - (step / EPS_DECAY_STEPS) * (EPS_START - EPS_END))

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.tensor(np.array(obs), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_t).argmax().item()

        # Step in Environment
        next_obs, reward, done, truncated, _ = env.step(action)

        # Store transition — MiniGrid rewards are already in (0, 1) so no clipping needed
        memory.append((obs, action, reward, next_obs, done or truncated))
        obs = next_obs
        episode_reward += reward  # Track real reward for logging

        # Optimize every 4 steps — same as paper
        if len(memory) > LEARNING_STARTS:
            if step % 4 == 0:  # Only optimize every 4 steps to save time
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Cast to uint8 — NatureDQN normalizes internally
                states      = torch.tensor(np.array(states),      dtype=torch.uint8).to(device)
                actions     = torch.tensor(actions).unsqueeze(1).to(device)
                rewards     = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.uint8).to(device)
                dones       = torch.tensor(dones, dtype=torch.float32).to(device)

                # Current Q values
                current_q = policy_net(states).gather(1, actions).squeeze()

                # Target Q values (Bellman Equation)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + GAMMA * max_next_q * (1 - dones)

                # Huber loss — clips error to [-1, 1] per paper
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
            print(f"Step: {step} | Reward: {episode_reward} | Epsilon: {epsilon:.3f}")

            episode_rewards.append(episode_reward)
            writer.add_scalar("Charts/Episode_Reward", episode_reward, step)
            writer.add_scalar("Charts/Epsilon", epsilon, step)
            if len(episode_rewards) >= 100:
                mean_100 = np.mean(episode_rewards[-100:])
                writer.add_scalar("Charts/Mean100_Reward", mean_100, step)

            # Save checkpoint every 5000 steps
            if step > 0 and step % 5000 < 100:
                save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards, CHECKPOINT_PATH)
                torch.save(policy_net.state_dict(), f"dqn_minigrid_seed{SEED}_model.pth")

            obs, _ = env.reset()
            episode_reward = 0

    writer.close()

if __name__ == "__main__":
    train()