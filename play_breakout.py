import ale_py
import gymnasium as gym
import torch
import numpy as np
import time
from gymnasium.wrappers import RecordVideo
from wrappers.atari_wrapper import make_atari_env
from models.dqn_model import NatureDQN

gym.register_envs(ale_py)

def play(checkpoint_path="dqn_breakout_checkpoint.pth", episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_atari_env("ALE/Breakout-v5", render_mode="rgb_array")

    # records every episode to videos/ folder
    env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda e: True)

    model = NatureDQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for episode in range(episodes):
        obs, _ = env.reset()

        # Force to launch the ball
        obs, reward, done, truncated, _ = env.step(1)

        episode_reward = 0
        done = False

        step_count = 0
        while not done:
            state_t = torch.tensor(np.array(obs), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_t).argmax().item()

            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            step_count += 1

            if step_count % 100 == 0:
                print(f"Episode {episode + 1} | Step {step_count} | Reward: {episode_reward}")

        print(f"Episode {episode + 1} | Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    play()