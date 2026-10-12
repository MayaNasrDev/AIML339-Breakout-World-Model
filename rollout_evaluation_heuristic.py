import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


data_file = "Data/ALE_Breakout-v5_heuristic_transitions.npz"
model_file = "environment_model_heuristic.pth"
rollout_length = 3
validation_start_offset = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# same model as training, needed to load the saved weights
class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4, action_embedding_size=16):
        super().__init__()

        #taking the current frame, extracting features 
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.resize = nn.Upsample(size=(84, 84), mode="bilinear", align_corners=False)
        #reward for taking action
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        encoded_observation = self.encoder(observation)
        encoded_observation = encoded_observation.view(encoded_observation.size(0), -1)
        encoded_action = self.action_embedding(action)

        combined = torch.cat((encoded_observation, encoded_action), dim=1)
        combined = self.combined_layer(combined)

        predicted_reward = self.reward_head(combined)
        decoder_input = combined.view(-1, 64, 8, 8)
        predicted_next_observation = self.resize(self.decoder(decoder_input))

        return predicted_next_observation, predicted_reward


def prepare_frame(frame):
    # images are stored as 84x84x1, pytorch needs 1x1x84x84
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return torch.from_numpy(frame).unsqueeze(0).to(device)


def main():
    Dataset = np.load(data_file)

    # use the same last two segments as validation during training
    segment_end_indices = np.where(Dataset["segment_end"])[0]
    num_train_segments = int(len(segment_end_indices) * 0.8)
    split_index = segment_end_indices[num_train_segments - 1] + 1
    start_index = split_index + validation_start_offset

    if start_index + rollout_length > len(Dataset["actions"]):
        raise ValueError("Not enough transitions for this rollout")

    # don't cross into a new segment in the middle of a rollout
    for i in range(rollout_length - 1):
        if Dataset["segment_end"][start_index + i]:
            raise ValueError("Rollout crosses a segment boundary, change the offset")

    model = EnvironmentModel().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()

    current_prediction = prepare_frame(Dataset["observations"][start_index])
    real_frames = []
    predicted_frames = []
    frame_mses = []

    print("Rollout starting at transition:", start_index)

    with torch.no_grad():
        for step in range(rollout_length):
            index = start_index + step
            action = int(Dataset["actions"][index])
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)

            # after step 1 the model uses its own predicted frame
            predicted_next, predicted_reward = model(current_prediction, action_tensor)
            actual_next = prepare_frame(Dataset["next_observations"][index])
            frame_mse = torch.mean((predicted_next - actual_next) ** 2).item()
            actual_reward = float(Dataset["rewards"][index])

            real_frames.append(actual_next[0, 0].cpu().numpy())
            predicted_frames.append(predicted_next[0, 0].cpu().numpy())
            frame_mses.append(frame_mse)

            print(f"Step {step + 1} | Action: {action} | Frame MSE: {frame_mse:.6f} | "f"Actual reward: {actual_reward:.2f} | Predicted reward: {predicted_reward.item():.4f}")
            #predicted frame because errors build up since it is predicting from its own output
            current_prediction = predicted_next

    # show actual and predicted frames side by side
    fig, axes = plt.subplots(2, rollout_length, figsize=(rollout_length * 3.5, 6))

    for step in range(rollout_length):
        axes[0, step].imshow(real_frames[step], cmap="gray")
        axes[0, step].set_title(f"Actual step {step + 1}")
        axes[0, step].axis("off")

        axes[1, step].imshow(predicted_frames[step], cmap="gray")
        axes[1, step].set_title(f"Predicted step {step + 1}\nMSE: {frame_mses[step]:.6f}")
        axes[1, step].axis("off")

    plt.tight_layout()
    plt.savefig("heuristic_short_rollout.png")
    plt.show()


if __name__ == "__main__":
    main()
