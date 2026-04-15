import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from wrappers.atari_wrapper import make_atari_env
from dqn_model import NatureDQN

def train():
    # Hyperparameters
    ENV_NAME = "ALE/Breakout-v5"
    LR = 1e-4
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 20000
    LEARNING_STARTS = 10000
    TARGET_UPDATE_FREQ = 1000
    TOTAL_STEPS = 1000000
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 500000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Make environment
    env = make_atari_env(ENV_NAME, render_mode=None)
    
    policy_net = NatureDQN(env.action_space.n).to(device)
    target_net = NatureDQN(env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=REPLAY_SIZE)
    
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(TOTAL_STEPS):
        # Choose Action (Epsilon-Greedy Policy)
        epsilon = max(EPS_END, EPS_START - step / EPS_DECAY)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Convert to float and normalize (0-255 -> 0.0-1.0)
            state_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device) / 255.0
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

            # Optimization: Cast to float32 and normalize pixels
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device) / 255.0
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device) / 255.0
            dones = torch.tensor(dones, dtype=torch.uint8).to(device)

            # Current Q values
            current_q = policy_net(states).gather(1, actions).squeeze()
            
            # Target Q values (Bellman Equation)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * max_next_q * (1 - dones.float())

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Logging & Periodic Saving
        if done or truncated:
            print(f"Step: {step} | Reward: {episode_reward} | Epsilon: {epsilon:.2f}")
            
            # Save the model every 50k steps
            if step > 0 and step % 50000 < 100: 
                torch.save(policy_net.state_dict(), "dqn_breakout_checkpoint.pth")
                
            obs, _ = env.reset()
            episode_reward = 0

if __name__ == "__main__":
    train()