# AIML339 short rollout evaluation
# start with a real Breakout frame, then use the model's predictions for the next steps

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

file = "Data/ALE_Breakout-v5_transitions.npz"
model_file = "environment_model.pth"  # original model used for frame prediction
rollout_length = 3


class EnvironmentModel(nn.Module):
    # same architecture as training so the saved weights load
    def __init__(self, num_actions=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU()
        )

        self.action_embedding = nn.Embedding(num_actions, 16)
        self.fc = nn.Sequential(nn.Linear(64 * 8 * 8 + 16, 64 * 8 * 8), nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=6, stride=2), nn.Sigmoid()
        )

        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 16, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        visual_features = self.encoder(observation)
        visual_features = visual_features.view(visual_features.size(0), -1)
        action_features = self.action_embedding(action)
        combined_features = torch.cat((visual_features, action_features), dim=1)

        predicted_reward = self.reward_head(combined_features)
        decoded_features = self.fc(combined_features).view(-1, 64, 8, 8)
        predicted_next_observation = self.decoder(decoded_features)
        return predicted_next_observation, predicted_reward


def prepare_frame(frame):
    # normalise pixels and put channel first for pytorch
    return torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0


def main():
    dataset = np.load(file)
    model = EnvironmentModel()
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()

    # episode based validation split, leave room for rollout also like training
    num_transitions = len(dataset["actions"])
    episode_ends = dataset["terminated"] | dataset["truncated"]
    split_index = int(num_transitions * 0.8)

    while split_index < num_transitions - 1 and not episode_ends[split_index - 1]:
        split_index += 1

    # avoiding crossing episode boundaries for the rollout, find a valid start index
    start_index = split_index + 20
    while np.any(episode_ends[start_index:start_index + rollout_length]):
        start_index += 1

    print("Validation starts here->", split_index)
    print("Rollout starts here->", start_index)

    current_prediction = prepare_frame(dataset["observations"][start_index])
    real_frames = []
    predicted_frames = []
    actual_rewards = []
    actions_used = []
    mse_each_step = []

    with torch.no_grad():
        for step in range(rollout_length):
            index = start_index + step
            action_value = int(dataset["actions"][index])
            action = torch.tensor([action_value], dtype=torch.long)

            # not a real frame, but the model's prediction from the previous step
            predicted_next_observation, predicted_reward = model(current_prediction, action)
            actual_next_tensor = prepare_frame(dataset["next_observations"][index])
            frame_mse = torch.mean((predicted_next_observation - actual_next_tensor) ** 2).item()

            real_frames.append(actual_next_tensor[0, 0].numpy())
            predicted_frames.append(predicted_next_observation[0, 0].numpy())
            actual_rewards.append(dataset["rewards"][index])
            actions_used.append(action_value)
            mse_each_step.append(frame_mse)

            print(
                f"Step {step + 1} | Action: {action_value} | Frame MSE: {frame_mse:.6f} | "
                f"Actual reward: {actual_rewards[-1]:.2f} | Predicted reward: {predicted_reward.item():.4f}"
            )
            current_prediction = predicted_next_observation

    # comparing real frames with model generated
    fig, axes = plt.subplots(2, rollout_length, figsize=(rollout_length * 3, 6))

    for step in range(rollout_length):
        axes[0, step].imshow(real_frames[step], cmap="gray")
        axes[0, step].set_title(
            f"Actual step {step + 1}\nAction: {actions_used[step]} | Reward: {actual_rewards[step]:.2f}"
        )
        axes[0, step].axis("off")

        axes[1, step].imshow(predicted_frames[step], cmap="gray")
        axes[1, step].set_title(f"Predicted step {step + 1}\nMSE: {mse_each_step[step]:.6f}")
        axes[1, step].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
