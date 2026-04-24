import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.light_drqn_model import DRQN
from buffers.episode_buffer import EpisodeReplayBuffer
from wrappers.minigrid_wrapper import make_minigrid_env
from torch.utils.tensorboard import SummaryWriter
import os

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
    # LIGHT Hyperparameters for CPU Efficiency
    ENV_NAME = "MiniGrid-MemoryS7-v0"
    SEED = 42

    LR = 0.0003  # Slightly higher LR for faster convergence
    GAMMA = 0.99
    BATCH_SIZE = 16  # Reduced from 32 to speed up CPU backprop
    UPDATE_EVERY = 16 # Increased from 4: Collect more data per update

    REPLAY_EPISODES = 500
    SEQUENCE_LENGTH = 8
    HIDDEN_SIZE = 128

    MIN_EPISODES_TO_TRAIN = 10
    TARGET_UPDATE_FREQ = 1000

    TOTAL_STEPS = 500000
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY_STEPS = 150000 # Faster decay for faster learning

    CHECKPOINT_PATH = f"checkpoints/drqn_light_seed{SEED}.pth"

    # Force CPU for this light script
    device = torch.device("cpu")
    print(f"Training Light DRQN on: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Make environment
    env = make_minigrid_env(ENV_NAME)
    env.action_space.seed(SEED)

    policy_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH).to(device)
    target_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Adam is often faster at finding a solution on CPU than RMSProp
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    memory = EpisodeReplayBuffer(capacity=REPLAY_EPISODES, sequence_length=SEQUENCE_LENGTH)

    start_step, episode_rewards = load_checkpoint(policy_net, target_net, optimizer, CHECKPOINT_PATH)
    writer = SummaryWriter(log_dir=f"logs/drqn_light_seed{SEED}", purge_step=start_step)

    obs, _ = env.reset()
    episode_reward = 0
    hidden = policy_net.init_hidden(batch_size=1, device=device)

    for step in range(start_step, TOTAL_STEPS):
        epsilon = max(EPS_END, EPS_START - (step / EPS_DECAY_STEPS) * (EPS_START - EPS_END))

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.from_numpy(np.array(obs)).unsqueeze(0).to(device)
            with torch.inference_mode():
                q_values, hidden = policy_net(state_t, hidden)
                action = q_values.argmax().item()

        next_obs, reward, done, truncated, _ = env.step(action)
        memory.push_transition(obs, action, reward, next_obs, done or truncated)
        obs = next_obs
        episode_reward += reward

        # Update every 16 steps
        if memory.ready(MIN_EPISODES_TO_TRAIN) and step % UPDATE_EVERY == 0:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            # Fast conversion
            states      = torch.from_numpy(states).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            actions     = torch.from_numpy(actions).long().to(device)
            rewards     = torch.from_numpy(rewards).float().to(device)
            dones       = torch.from_numpy(dones).float().to(device)

            q_values, _ = policy_net(states)
            actions_last = actions[:, -1].unsqueeze(1)
            current_q = q_values.gather(1, actions_last).squeeze(1)

            with torch.no_grad():
                next_q_values, _ = target_net(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards[:, -1] + GAMMA * max_next_q * (1 - dones[:, -1])

            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Losses/TD_Loss", loss.item(), step)

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done or truncated:
            if step % 100 == 0:
                print(f"Step: {step} | Reward: {episode_reward:.2f} | Eps: {epsilon:.2f}")

            episode_rewards.append(episode_reward)
            writer.add_scalar("Charts/Episode_Reward", episode_reward, step)
            
            obs, _ = env.reset()
            episode_reward = 0
            hidden = policy_net.init_hidden(batch_size=1, device=device)

            if step > 0 and step % 10000 < 500:
                save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards, CHECKPOINT_PATH)

    writer.close()

if __name__ == "__main__":
    train()