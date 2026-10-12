"""One-step learned Breakout environment using the heuristic world model."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn


data_file = "Data/ALE_Breakout-v5_heuristic_transitions.npz"
model_file = "environment_model_heuristic.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# same architecture as heuristic training just to load saved weights
class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4, action_embedding_size=16):
        super().__init__()

        # extracts features from the current game frame
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.action_embedding = nn.Embedding(num_actions, action_embedding_size)
        self.combined_layer = nn.Sequential(
            nn.Linear(64 * 10 * 10 + action_embedding_size, 64 * 8 * 8),
            nn.ReLU()
        )

        # predicts the next frame
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.resize = nn.Upsample(size=(84, 84), mode="bilinear", align_corners=False)

        # predicts reward from the same combined features
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        encoded = self.encoder(observation)
        encoded = encoded.view(encoded.size(0), -1)
        action_features = self.action_embedding(action)
        combined = self.combined_layer(torch.cat((encoded, action_features), dim=1))

        predicted_reward = self.reward_head(combined)
        predicted_frame = self.decoder(combined.view(-1, 64, 8, 8))
        predicted_frame = self.resize(predicted_frame)
        return predicted_frame, predicted_reward


class LearnedBreakoutEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        self.dataset = np.load(data_file)

        # only first 8 segments last 2 for validation.
        segment_ends = np.where(self.dataset["segment_end"])[0]
        num_train_segments = int(len(segment_ends) * 0.8)
        self.train_end = segment_ends[num_train_segments - 1] + 1

        self.model = EnvironmentModel().to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        self.model.eval()
        self.current_observation = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # rollout from a real frame in the training set, but not too close to the end of the training data
        start_index = self.np_random.integers(0, self.train_end)
        self.current_observation = self.dataset["observations"][start_index].copy()
        return self.current_observation, {"start_index": int(start_index)}

    def step(self, action):
        # convert image from 84x84x1 uint8 to a pytorch batch
        frame = self.current_observation.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)

        with torch.no_grad():
            predicted_frame, predicted_reward = self.model(frame_tensor, action_tensor)

        # change the predicted frame back to the Gymnasium image format
        frame = predicted_frame[0, 0].cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        self.current_observation = np.expand_dims(frame, axis=-1)
        reward = float(predicted_reward.item())

        # only one model generated step before resetting to a real frame otherwise errors accumulate too much
        terminated = False
        truncated = True
        return self.current_observation, reward, terminated, truncated, {"model_generated": True}


# action 0 = NOOP, 1 = FIRE, 2 = RIGHT, 3 = LEFT
def main():
    env = LearnedBreakoutEnv()
    observation, info = env.reset(seed=350)
    print("Starting index:", info["start_index"])
    print("Observation shape:", observation.shape)

    action = 2  
    next_observation, reward, terminated, truncated, info = env.step(action)
    print("Action:", action)
    print("Next observation shape:", next_observation.shape)
    print("Predicted reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Model generated:", info["model_generated"])
    env.close()


if __name__ == "__main__":
    main()
