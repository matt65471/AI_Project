import torch
import random
import numpy as np
from wrappers.minigrid_wrapper import make_minigrid_env
from models.dqn_model import NatureDQN

torch.backends.nnpack.set_flags(False)

def play(checkpoint_path="dqn_minigrid_seed21_model.pth", episodes=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MiniGrid environment (with symbolic observations)
    env = make_minigrid_env("MiniGrid-MemoryS7-v0", render_mode="human")  # Change to 'human' for real-time rendering

    # Load the pre-trained model
    model = NatureDQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Run for a specified number of episodes
    for episode in range(episodes):
        random_seed = random.randint(0, 1000000)
        obs, _ = env.reset(seed=random_seed)
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # Convert observation to tensor for the model
            state_t = torch.tensor(np.array(obs), dtype=torch.uint8).unsqueeze(0).to(device)
            
            # Perform action selection (inference)
            with torch.no_grad():
                action = model(state_t).argmax().item()

            # Step through the environment
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            step_count += 1

            # Render the environment in real-time
            env.render()

            if step_count % 100 == 0:
                print(f"Episode {episode + 1} | Step {step_count} | Reward: {episode_reward}")

        print(f"Episode {episode + 1} | Total Reward: {episode_reward}")

    # Close the environment after all episodes are finished
    env.close()

if __name__ == "__main__":
    play()