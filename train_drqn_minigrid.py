import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.drqn_model import DRQN
from buffers.episode_buffer import EpisodeReplayBuffer
from wrappers.minigrid_wrapper import make_minigrid_env
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
    # DRQN Hyperparameters for MiniGrid MemoryEnv
    ENV_NAME = "MiniGrid-MemoryS7-v0"
    SEED = 42

    LR = 0.00025
    GAMMA = 0.99
    BATCH_SIZE = 32

    # Number of episodes in replay buffer
    REPLAY_EPISODES = 500

    # Sequence length for LSTM training
    SEQUENCE_LENGTH = 8

    # LSTM hidden size
    HIDDEN_SIZE = 512

    # Start training after 10 episodes
    MIN_EPISODES_TO_TRAIN = 10

    # Target network update frequency
    TARGET_UPDATE_FREQ = 1000

    TOTAL_STEPS = 500000
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY_STEPS = 250000

    CHECKPOINT_PATH = f"checkpoints/drqn_minigrid_seed{SEED}_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DRQN on: {device} | Seed: {SEED}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Make environment
    env = make_minigrid_env(ENV_NAME)
    env.action_space.seed(SEED)

    # DRQN instead of NatureDQN — this is the required algorithmic change
    policy_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH).to(device)
    target_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is never trained directly

    # RMSProp — same as paper
    optimizer = optim.RMSprop(
        policy_net.parameters(),
        lr=LR,
        alpha=0.95,
        eps=0.01,
        momentum=0.95,
        centered=True
    )

    # Episode-based replay buffer — required for DRQN
    memory = EpisodeReplayBuffer(
        capacity=REPLAY_EPISODES,
        sequence_length=SEQUENCE_LENGTH
    )

    # Load checkpoint if it exists
    start_step, episode_rewards = load_checkpoint(policy_net, target_net, optimizer, CHECKPOINT_PATH)

    writer = SummaryWriter(log_dir=f"logs/drqn_minigrid_seed{SEED}", purge_step=start_step)

    obs, _ = env.reset()
    episode_reward = 0

    # LSTM hidden state — reset at each episode start
    hidden = policy_net.init_hidden(batch_size=1, device=device)

    for step in range(start_step, TOTAL_STEPS):
        # Choose Action (Epsilon-Greedy Policy)
        epsilon = max(EPS_END, EPS_START - (step / EPS_DECAY_STEPS) * (EPS_START - EPS_END))

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Pass hidden state so LSTM remembers across steps
            state_t = torch.tensor(np.array(obs), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, hidden = policy_net(state_t, hidden)
                action = q_values.argmax().item()

        # Step in Environment
        next_obs, reward, done, truncated, _ = env.step(action)

        # Store transition in episode buffer
        memory.push_transition(obs, action, reward, next_obs, done or truncated)
        obs = next_obs
        episode_reward += reward

        # Optimize every 4 steps once buffer has enough episodes
        if memory.ready(MIN_EPISODES_TO_TRAIN) and step % 4 == 0:
            # Sample sequences — not random transitions
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            # Convert to tensors — shape (batch, seq, 4, 84, 84)
            states      = torch.tensor(states,      dtype=torch.uint8).to(device)
            next_states = torch.tensor(next_states, dtype=torch.uint8).to(device)
            actions     = torch.tensor(actions,     dtype=torch.long).to(device)
            rewards     = torch.tensor(rewards,     dtype=torch.float32).to(device)
            dones       = torch.tensor(dones,       dtype=torch.float32).to(device)

            # Current Q values — LSTM processes full sequence
            q_values, _ = policy_net(states)

            # Only use the last action in the sequence for loss
            actions_last = actions[:, -1].unsqueeze(1)
            current_q = q_values.gather(1, actions_last).squeeze(1)

            # Target Q values
            with torch.no_grad():
                next_q_values, _ = target_net(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards[:, -1] + GAMMA * max_next_q * (1 - dones[:, -1])

            # Huber loss — clips error to [-1, 1] per paper
            loss = nn.SmoothL1Loss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Losses/TD_Loss", loss.item(), step)

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Reset hidden state and log at episode end
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
                torch.save(policy_net.state_dict(), f"drqn_minigrid_seed{SEED}_model.pth")

            # Reset environment and LSTM hidden state for new episode
            obs, _ = env.reset()
            episode_reward = 0
            hidden = policy_net.init_hidden(batch_size=1, device=device)  # ← key difference from DQN

    writer.close()

if __name__ == "__main__":
    train()