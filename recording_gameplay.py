import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
gym.register_envs(ale_py)

#which agent to record either ppo_baseline or ppo_model_ass
model_file = "ppo_baseline" 

env = gym.make("ALE/Breakout-v5", render_mode = "rgb_array", obs_type="grayscale", frameskip=1, repeat_action_probability=0.0)

env = gym.wrappers.ResizeObservation(env, (84,84))

# gameplay into mp4, make sure name_prefix is correct
env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="baseline", episode_trigger = lambda episode:True)

#identfical observation shape to training
class AddChannelDImension(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84,84,1), dtype=env.observation_space.dtype)
    def observation(self, observation): return observation[:, :,None]
    
env = AddChannelDImension(env)
model = PPO.load(model_file)
observation,info = env.reset(seed=350)

#launching ball before agent begings
observation, reward, terminated, truncated, info = env.step(1)
for step in range(3000): 
    action, _ = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = env.step(int(action))
    
    if terminated or truncated: 
        break

env.close()
print("vid saved")