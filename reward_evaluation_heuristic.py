"""Check reward predictions from the heuristic Breakout environment model."""

import numpy as np
import torch
import torch.nn as nn

# same dataset and saved model used in heuristic training
data_file = 'Data/ALE_Breakout-v5_heuristic_transitions.npz'
model_file = 'environment_model_heuristic.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model architecture must match training so the saved weights load
class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4, action_embedding_size=16):
        super().__init__()

        # encoder extracts features from the current game frame
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )

        self.action_embedding = nn.Embedding(num_actions, action_embedding_size)

        # combine frame + action features
        self.combined_layer = nn.Sequential(
            nn.Linear(64 * 10 * 10 + action_embedding_size, 64 * 8 * 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )
        self.resize = nn.Upsample(size=(84, 84), mode='bilinear', align_corners=False)

        # seperately predicts reward
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        encoded_observation = self.encoder(observation)
        encoded_observation = encoded_observation.view(encoded_observation.size(0), -1)
        encoded_action = self.action_embedding(action)

        combined = torch.cat((encoded_observation, encoded_action), dim=1)
        combined = self.combined_layer(combined)

        predicted_reward = self.reward_head(combined)
        decoder_input = combined.view(-1, 64, 8, 8)
        predicted_next_observation = self.decoder(decoder_input)
        predicted_next_observation = self.resize(predicted_next_observation)

        return predicted_next_observation, predicted_reward


def main():
    Dataset = np.load(data_file)

    # last 20% of gameplay segments used for validation, leave room for rollouts
    segment_end_indices = np.where(Dataset['segment_end'])[0]
    num_segments = len(segment_end_indices)
    if num_segments < 2:
        raise ValueError('not enough gameplay segment in dataset!')

    num_train_segments = int(num_segments * 0.8)
    split_index = segment_end_indices[num_train_segments - 1] + 1

    validation_observations = Dataset['observations'][split_index:]
    validation_actions = Dataset['actions'][split_index:]
    validation_rewards = Dataset['rewards'][split_index:]

    print('Validation transitions:', len(validation_actions))
    print('Validation non-zero rewards:', np.count_nonzero(validation_rewards))

    # load the existing model, no retraining
    model = EnvironmentModel().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()

    # same image preprocessing as during training
    observations = validation_observations.astype(np.float32) / 255.0
    observations = np.transpose(observations, (0, 3, 1, 2))
    observations_tensor = torch.from_numpy(observations).to(device)
    actions_tensor = torch.from_numpy(validation_actions.astype(np.int64)).to(device)

    predicted_rewards_list = []
    evaluation_batch_size = 64

    # predict rewards in batches, rather than passing all frames at once
    with torch.no_grad():
        for start in range(0, len(observations_tensor), evaluation_batch_size):
            end = start + evaluation_batch_size
            observation_batch = observations_tensor[start:end]
            action_batch = actions_tensor[start:end]

            _, predicted_reward = model(observation_batch, action_batch)
            predicted_rewards_list.append(predicted_reward.squeeze(1).cpu().numpy())

    predicted_rewards = np.concatenate(predicted_rewards_list)
    actual_rewards = validation_rewards.astype(np.float32)
    zero_predictions = np.zeros_like(actual_rewards)

    # compare model errors against always predicting reward = 0
    model_mse = np.mean((predicted_rewards - actual_rewards) ** 2)
    zero_baseline_mse = np.mean((zero_predictions - actual_rewards) ** 2)

    # split results because most Breakout transitions have zero reward
    zero_mask = actual_rewards == 0
    nonzero_mask = actual_rewards != 0

    model_zero_mse = np.mean((predicted_rewards[zero_mask] - actual_rewards[zero_mask]) ** 2)
    baseline_zero_mse = np.mean((zero_predictions[zero_mask] - actual_rewards[zero_mask]) ** 2)

    if np.any(nonzero_mask):
        model_nonzero_mse = np.mean((predicted_rewards[nonzero_mask] - actual_rewards[nonzero_mask]) ** 2)
        baseline_nonzero_mse = np.mean((zero_predictions[nonzero_mask] - actual_rewards[nonzero_mask]) ** 2)
    else:
        model_nonzero_mse = np.nan
        baseline_nonzero_mse = np.nan

    print('\n-- HEURISTIC REWARD PREDICTION RESULTS --')
    print(f'Overall model MSE: {model_mse:.6f}')
    print(f'Overall always zero MSE: {zero_baseline_mse:.6f}')

    print(f'\nZero reward transitions: {np.sum(zero_mask)}')
    print(f'Model MSE: {model_zero_mse:.6f}')
    print(f'Always zero MSE: {baseline_zero_mse:.6f}')

    print(f'\nNon zero reward transitions: {np.sum(nonzero_mask)}')
    print(f'Model MSE: {model_nonzero_mse:.6f}')
    print(f'Always zero MSE: {baseline_nonzero_mse:.6f}')

    # does model predict zero reward when actual reward is zero?
    zero_reward_predictions = predicted_rewards[zero_mask]
    if len(zero_reward_predictions) > 0:
        print('\nPredictions when actual reward = 0')
        print(f'Average: {np.mean(zero_reward_predictions):.6f}')
        print(f'Maximum: {np.max(zero_reward_predictions):.6f}')
        print(f'Minimum: {np.min(zero_reward_predictions):.6f}')

    # show the actual vs predicted for the rare non zero rewards
    print('\nActual vs predicted non zero rewards')
    nonzero_indices = np.where(nonzero_mask)[0]
    for index in nonzero_indices:
        print(f'Index {index} | Actual: {actual_rewards[index]:.1f} | Predicted: {predicted_rewards[index]:.6f}')

    if len(nonzero_indices) > 0:
        print(f'Avg predicted non zero reward: {np.mean(predicted_rewards[nonzero_mask]):.6f}')
        print(f'Avg actual non zero reward: {np.mean(actual_rewards[nonzero_mask]):.6f}')


if __name__ == '__main__':
    main()
