import ale_py  # registers ALE/* envs (required with gymnasium 1.x)
import gymnasium as gym

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()