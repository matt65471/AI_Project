import gymnasium as gym

def make_atari_env(env_id, render_mode=None):
    # Base environment
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    
    # Preprocessing
    env = gym.wrappers.AtariPreprocessing(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=True, 
        grayscale_obs=True
    )
    
    # Frame Stacking
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    return env