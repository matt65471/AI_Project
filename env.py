import ale_py  # registers ALE/* envs (required with gymnasium 1.x)
import gymnasium as gym

from atari_preprocessing import make_atari_dqn_env

# Raw ALE is 210×160 RGB; wrapper: max over last two frames → Y → 84×84 → stack m=4 → (4, 84, 84)
env = make_atari_dqn_env("ALE/Pong-v5", render_mode="human", frame_stack=4)
obs, info = env.reset()
assert obs.shape == (4, 84, 84)

for _ in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()
