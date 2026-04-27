import torch
import minigrid
import cv2
import numpy as np
from gymnasium.wrappers import RecordVideo
from wrappers.light_minigrid_wrapper import make_minigrid_env
from models.light_drqn_model import DRQN  # Ensure this is the correct model you are using

torch.backends.nnpack.set_flags(False)

def play(checkpoint_path="drqn_light_seed50_model.pth", episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MiniGrid environment
    env = make_minigrid_env("MiniGrid-MemoryS7-v0", render_mode="rgb_array")

    # Record videos of each episode, saved in the "videos/minigrid/" folder
    env = RecordVideo(env, video_folder="videos/light_minigrid/", episode_trigger=lambda e: True)

    # Initialize the DRQN model
    model = DRQN(env.action_space.n, hidden_size=128, sequence_length=8, obs_shape=env.observation_space.shape).to(device)

    # Load the trained model's state_dict
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set to evaluation mode

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        # Initialize the LSTM hidden state for each episode
        hidden = model.init_hidden(batch_size=1, device=device)

        while not done:
            symbolic_view = obs[0]
            view_scaled = (symbolic_view * 25).astype(np.uint8)
            view_large = cv2.resize(view_scaled, (280, 280), interpolation = cv2.INTER_NEAREST)

            cv2.imshow("AI Vision", view_large)
            cv2.waitKey(10000)

            # Convert observation to tensor (no need to slice here)
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)

            # Perform inference using the model to select the action
            with torch.no_grad():
                q_values, hidden = model(obs_tensor, hidden)
                action = q_values.argmax().item()

            # Take the action in the environment
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            step_count += 1

            if step_count % 100 == 0:
                print(f"Episode {episode + 1} | Step {step_count} | Reward: {episode_reward}")

        print(f"Episode {episode + 1} | Total Reward: {episode_reward}")

    # Close the environment after all episodes
    env.close()

if __name__ == "__main__":
    play()