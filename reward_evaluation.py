"""Evaluate reward predictions from the random-data Breakout model."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

file = 'Data/ALE_Breakout-v5_transitions.npz'
model_file = 'environment_model_weighted.pth'
batch_size = 32


class BreakoutDataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        # Same image preprocessing used for training
        observation = torch.tensor(self.observations[index], dtype=torch.float32) / 255.0
        next_observation = torch.tensor(self.next_observations[index], dtype=torch.float32) / 255.0
        action = torch.tensor(self.actions[index], dtype=torch.long)
        reward = torch.tensor(self.rewards[index], dtype=torch.float32)
        return observation.permute(2, 0, 1), action, reward, next_observation.permute(2, 0, 1)

#using the same validation data as training 4 fair evaluation, moving the split 2 an episode bound. as well.
def split_dataset(dataset, train_ratio=0.8):
    # Match the training split and keep complete episodes together
    num_transitions = len(dataset['actions'])
    split_index = int(num_transitions * train_ratio)
    episode_ends = dataset['terminated'] | dataset['truncated']

    while split_index < num_transitions - 1 and not episode_ends[split_index - 1]:
        split_index += 1

    return BreakoutDataset(
        dataset['observations'][split_index:],
        dataset['actions'][split_index:],
        dataset['rewards'][split_index:],
        dataset['next_observations'][split_index:]
    )


# Must match the original model architecture so the checkpoint loads
class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        #the CNN extracts features from the current breakout frame, reducing img -> smaller rep
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU()
        )
        #next frame dependent on which action was taken + action num -> learned vect.
        self.action_embedding = nn.Embedding(num_actions, 16)
        self.fc = nn.Sequential(nn.Linear(64 * 8 * 8 + 16, 64 * 8 * 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=6, stride=2), nn.Sigmoid()
        )
        #single reward value prediction from frame + action feat. 
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 16, 128), nn.ReLU(), nn.Linear(128, 1)
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


def main():
    dataset = np.load(file)
    validation_data = split_dataset(dataset)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model = EnvironmentModel()
    #load weights from model no training happens
    model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
    model.eval()

    print('Validation transitions:', len(validation_data))

    model_squared_errors = []
    zero_baseline_squared_errors = []
    zero_reward_errors = []
    nonzero_reward_errors = []
    zero_baseline_zero_errors = []
    zero_baseline_nonzero_errors = []
    nonzero_actual_rewards = []
    nonzero_predicted_rewards = []
    zero_predicted_rewards = []

    # Compare model reward predictions against always predicting zero
    with torch.no_grad():
        for observation, action, reward, next_observation in validation_loader:
            # reward prediction only no need for predicted next frame
            _, predicted_reward = model(observation, action)
            reward = reward.unsqueeze(1)
            #squared diff between predicted and actual reward
            model_error = (predicted_reward - reward) ** 2
            #simple baseline that always guesses 0 cuz most transition have no reward
            zero_baseline_error = (torch.zeros_like(reward) - reward) ** 2

            model_squared_errors.extend(model_error.flatten().tolist())
            zero_baseline_squared_errors.extend(zero_baseline_error.flatten().tolist())

            # Separate the common zero rewards from the rare non-zero rewards
            zero_mask = reward == 0
            nonzero_mask = reward != 0

            if zero_mask.any():
                zero_reward_errors.extend(model_error[zero_mask].tolist())
                zero_baseline_zero_errors.extend(zero_baseline_error[zero_mask].tolist())
                zero_predicted_rewards.extend(predicted_reward[zero_mask].tolist())

            if nonzero_mask.any():
                nonzero_reward_errors.extend(model_error[nonzero_mask].tolist())
                zero_baseline_nonzero_errors.extend(zero_baseline_error[nonzero_mask].tolist())
                nonzero_actual_rewards.extend(reward[nonzero_mask].tolist())
                nonzero_predicted_rewards.extend(predicted_reward[nonzero_mask].tolist())

    #comparing avg error across all transitions, checking actual prediction for rare non-zero rewards
    print('\n--REWARD PREDICTION RES--')
    print('Overall model MSE:', round(np.mean(model_squared_errors), 6))
    print('Overall always zero MSE:', round(np.mean(zero_baseline_squared_errors), 6))

    print('\nZero reward transitions:', len(zero_reward_errors))
    if zero_reward_errors:
        print('Model MSE:', round(np.mean(zero_reward_errors), 6))
        print('Always-zero MSE:', round(np.mean(zero_baseline_zero_errors), 6))
        print('Average prediction:', round(np.mean(zero_predicted_rewards), 6))
        print('Largest prediction:', round(np.max(zero_predicted_rewards), 6))
        print('Smallest prediction:', round(np.min(zero_predicted_rewards), 6))

    print('\nNon-zero reward transitions:', len(nonzero_reward_errors))
    if nonzero_reward_errors:
        print('Model MSE:', round(np.mean(nonzero_reward_errors), 6))
        print('Always-zero MSE:', round(np.mean(zero_baseline_nonzero_errors), 6))
        # useful to see cuz few rewards
        for actual, predicted in zip(nonzero_actual_rewards, nonzero_predicted_rewards):
            print(f'Actual: {actual:.2f} | Predicted: {predicted:.4f}')
    else:
        print('No non-zero rewards in validation data')


if __name__ == '__main__':
    main()
