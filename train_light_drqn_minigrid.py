import minigrid
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.light_drqn_model import DRQN
from buffers.episode_buffer import EpisodeReplayBuffer
from wrappers.light_minigrid_wrapper import make_minigrid_env
from torch.utils.tensorboard import SummaryWriter
import os

torch.backends.nnpack.set_flags(False)  # Disable NNPACK to avoid potential issues on some platforms

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
    # DRQN Hyperparameters for MiniGrid MemoryEnv (Light CPU version)
    ENV_NAME = "MiniGrid-MemoryS7-v0"
    SEED = 40

    # Slightly higher LR for faster convergence on CPU
    LR = 0.00005

    # Discount factor
    GAMMA = 0.99

    # Reduced batch size for CPU speed
    BATCH_SIZE = 32

    # Number of episodes in replay buffer
    REPLAY_EPISODES = 1000

    # Shorter sequences = faster LSTM on CPU
    SEQUENCE_LENGTH = 16

    # Smaller LSTM = faster CPU training
    HIDDEN_SIZE = 128

    # Start training after 100 episodes
    MIN_EPISODES_TO_TRAIN = 100

    # Target network update frequency
    TARGET_UPDATE_FREQ = 10000

    TOTAL_STEPS = 2000000
    EPS_START = 1
    EPS_END = 0.05

    # Faster decay for faster learning signal
    EPS_DECAY_STEPS = 100000

    # Update every 8 steps instead of 4 — biggest CPU speed gain
    UPDATE_EVERY = 8
    SAVE_INTERVAL = 5000

    CHECKPOINT_PATH = f"checkpoints/drqn_light_seed{SEED}.pth"

    device = torch.device("cpu")
    print(f"Training Light DRQN on: {device} | Seed: {SEED}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Make environment — uses symbolic obs instead of pixels
    env = make_minigrid_env(ENV_NAME)
    env.action_space.seed(SEED)

    # Get obs shape from env dynamically
    obs_shape = env.observation_space.shape  # (3, H, W)
    print(f"Observation shape: {obs_shape}")

    # Light DRQN — smaller CNN + smaller LSTM
    policy_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH, obs_shape=obs_shape).to(device)
    target_net = DRQN(env.action_space.n, hidden_size=HIDDEN_SIZE, sequence_length=SEQUENCE_LENGTH, obs_shape=obs_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is never trained directly

    # RMSProp
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

    writer = SummaryWriter(log_dir=f"logs/drqn_light_seed{SEED}", purge_step=start_step)

    obs, _ = env.reset()
    episode_reward = 0
    shaped_episode_reward = 0
    episode_len = 0

    # LSTM hidden state — reset at each episode start
    hidden = policy_net.init_hidden(batch_size=1, device=device)
    last_saved_step = start_step

    for step in range(start_step, TOTAL_STEPS):
        if step == 0:
            obs_arr = np.array(obs)
            print(f"Obs shape: {obs_arr.shape}")
            print(f"Obs min: {obs_arr.min()} max: {obs_arr.max()}")
            print(f"Obs unique values: {np.unique(obs_arr)}")

        # Choose Action (Epsilon-Greedy Policy)
        epsilon = max(EPS_END, EPS_START - (step / EPS_DECAY_STEPS) * (EPS_START - EPS_END))

        # Pass hidden state so LSTM remembers across steps
        state_t = torch.from_numpy(np.array(obs)).unsqueeze(0).to(device)

        with torch.inference_mode():
                q_values, hidden = policy_net(state_t, hidden)
                action = q_values.argmax().item()

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            hidden = (hidden[0].detach(), hidden[1].detach())

            if step % 1000 == 0:
                print(f"Q-values: {q_values.numpy()} | Action: {action}")

        # Step in Environment
        next_obs, reward, done, truncated, _ = env.step(action)

        shaped_reward = reward
        if truncated and reward == 0:
            shaped_reward = -0.5
        elif done and reward == 0:
            shaped_reward = -1.0
        elif reward > 0:
            shaped_reward = 1.0

        # Store transition in episode buffer
        memory.push_transition(obs, action, shaped_reward, next_obs, done or truncated)

        obs = next_obs
        shaped_episode_reward += shaped_reward
        episode_reward += reward
        episode_len += 1

        # Optimize every x steps once buffer has enough episodes
        if memory.ready(MIN_EPISODES_TO_TRAIN) and step % UPDATE_EVERY == 0:
            # Sample sequences — not random transitions
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            # Fast conversion
            states      = torch.from_numpy(states).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            actions     = torch.from_numpy(actions).long().to(device)
            rewards     = torch.from_numpy(rewards).float().to(device)
            dones       = torch.from_numpy(dones).float().to(device)

            # Current Q values — LSTM processes full sequence
            train_hidden = policy_net.init_hidden(batch_size=BATCH_SIZE, device=device)
            q_values, _ = policy_net(states, train_hidden)

            # Only use last action in sequence for loss
            actions_last = actions[:, -1].unsqueeze(1)
            current_q = q_values.gather(1, actions_last).squeeze(1)

            # Target Q values (Bellman Equation)
            with torch.no_grad():
                target_hidden = target_net.init_hidden(batch_size=BATCH_SIZE, device=device)
                next_q_values, _ = target_net(next_states, target_hidden)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards[:, -1] + GAMMA * max_next_q * (1 - dones[:, -1])

            # Huber loss — clips error to [-1, 1] per paper
            loss = nn.SmoothL1Loss()(current_q, target_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            optimizer.step()

            writer.add_scalar("Losses/TD_Loss", loss.item(), step)

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (step - last_saved_step) >= SAVE_INTERVAL: 
            save_checkpoint(step, policy_net, target_net, optimizer, episode_rewards, CHECKPOINT_PATH) 
            torch.save(policy_net.state_dict(), f"drqn_light_seed{SEED}_model.pth") 
            last_saved_step = step 

        # Reset hidden state and log at episode end
        if done or truncated:
            print(f"Episode length: {episode_len} | Sequence length: {memory.sequence_length}")
            print(f"Step: {step} | Env Reward: {episode_reward:.2f} | Shaped Reward: {shaped_episode_reward:.2f} | Epsilon: {epsilon:.3f} | Buffer: {len(memory)} episodes")

            episode_rewards.append(episode_reward)
            writer.add_scalar("Charts/Episode_Reward", episode_reward, step)
            writer.add_scalar("Charts/Episode_Reward_Shaped", shaped_episode_reward, step)
            writer.add_scalar("Charts/Epsilon", epsilon, step)
            writer.add_scalar("Charts/Episode_Length", episode_len, step)
            if len(episode_rewards) >= 100:
                mean_100 = np.mean(episode_rewards[-100:])
                writer.add_scalar("Charts/Mean100_Reward", mean_100, step)

            # Reset environment and LSTM hidden state for new episode
            obs, _ = env.reset()
            episode_reward = 0
            episode_len = 0
            shaped_episode_reward = 0
            hidden = policy_net.init_hidden(batch_size=1, device=device)

    writer.close()

if __name__ == "__main__":
    train()