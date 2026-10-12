import csv
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


# 10k real steps already used for world model, 90k left for PPO
ENV_NAME = "ALE/Breakout-v5"
num_real_envs = 4
num_model_envs = 1
model_steps = 10000
ppo_steps = 90000
seed = 350

data_file = "Data/ALE_Breakout-v5_heuristic_transitions.npz"
environment_model_file = "environment_model_heuristic.pth"
model_path = "ppo_model_ass"
log_file = "ppo_model_ass_training.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gym.register_envs(ale_py)


# same architecture as the trained heuristic world model
class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4, action_embedding_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.action_embedding = nn.Embedding(num_actions, action_embedding_size)
        self.combined_layer = nn.Sequential(
            nn.Linear(64 * 10 * 10 + action_embedding_size, 64 * 8 * 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
            nn.Upsample(size=(84, 84), mode="bilinear", align_corners=False)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        frame = self.encoder(observation)
        frame = frame.view(frame.size(0), -1)
        action_features = self.action_embedding(action)
        combined = self.combined_layer(torch.cat((frame, action_features), dim=1))
        reward = self.reward_head(combined)
        next_frame = self.decoder(combined.view(-1, 64, 8, 8))
        return next_frame, reward


class AddChannelDimension(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        return observation[:, :, None]


def make_breakout_env(env_seed):
    def _init():
        env = gym.make(ENV_NAME, obs_type="grayscale", frameskip=1,
        repeat_action_probability=0.0)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = AddChannelDimension(env)
        env.reset(seed=env_seed)
        return env
    return _init


class LearnedBreakoutEnvironment(gym.Env):
    def __init__(self, data_file, model_file):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        dataset = np.load(data_file)
        self.observations = dataset["observations"]
        # only the first 8 heuristic segments were used for training
        segment_ends = np.where(dataset["segment_end"])[0]
        self.train_end = segment_ends[7] + 1

        self.environment_model = EnvironmentModel().to(device)
        self.environment_model.load_state_dict(
            torch.load(model_file, map_location=device, weights_only=True)
        )
        self.environment_model.eval()
        self.current_observation = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_index = self.np_random.integers(0, self.train_end)
        self.current_observation = self.observations[start_index].copy()
        return self.current_observation, {"model_generated": False,
                                "start_index": int(start_index)}

    def step(self, action):
        # convert frame to channel-first float tensor for the CNN
        frame = self.current_observation.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame_tensor = torch.tensor(frame, dtype=torch.float32,
                                    device=device).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)

        with torch.no_grad():
            next_frame, reward = self.environment_model(frame_tensor, action_tensor)

        # convert predicted frame back to the Gymnasium image format
        next_frame = next_frame.squeeze(0).cpu().numpy()
        next_frame = np.transpose(next_frame, (1, 2, 0))
        next_frame = np.clip(next_frame * 255.0, 0, 255).astype(np.uint8)
        self.current_observation = next_frame

        # one synthetic step only, so prediction errors don't keep building up
        return next_frame, float(reward.item()), False, True, {"model_generated": True}


def make_model_env():
    def _init():
        return LearnedBreakoutEnvironment(data_file, environment_model_file)
    return _init


class InteractionStepCB(BaseCallback):
    def __init__(self, real_steps_budget, log_file):
        super().__init__()
        self.real_steps_budget = real_steps_budget
        self.log_file = log_file
        self.real_steps = 0
        self.model_generated_steps = 0

    def _on_training_start(self):
        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ppo_steps", "ppo_real_steps", "model_generated_steps","model_training_real_steps", "total_real_steps"])

    def _on_step(self):
        # each vectorised step contains 4 real and 1 generated interaction
        self.real_steps += num_real_envs
        self.model_generated_steps += num_model_envs
        total_real_steps = model_steps + self.real_steps

        if self.real_steps % 1000 == 0:
            with open(self.log_file, "a", newline="") as file:
                csv.writer(file).writerow([
                    self.num_timesteps, self.real_steps, self.model_generated_steps,
                    model_steps, total_real_steps
                ])

        return self.real_steps < self.real_steps_budget


envs = [make_breakout_env(seed + i) for i in range(num_real_envs)]
envs.append(make_model_env())
env = VecMonitor(DummyVecEnv(envs))

model = PPO(
    "CnnPolicy", env, learning_rate=0.00025, n_steps=128,
    batch_size=256, gamma=0.99, verbose=1, seed=seed, device="auto"
)
callback = InteractionStepCB(real_steps_budget=ppo_steps, log_file=log_file)

# PPO counts all five envs, but the callback stops based on real steps
total_ppo_steps = int(ppo_steps * ((num_real_envs + num_model_envs) / num_real_envs))
model.learn(total_timesteps=total_ppo_steps + len(envs), callback=callback)
model.save(model_path)

print("Real PPO interactions:", callback.real_steps)
print("Model generated interactions:", callback.model_generated_steps)
print("Total real interactions (including model data):", model_steps + callback.real_steps)
print("Saved agent to:", model_path)
env.close()
