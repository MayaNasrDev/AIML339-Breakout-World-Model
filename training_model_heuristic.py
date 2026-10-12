"""Training the environment model using the heuristic Breakout dataset.
Takes the current observation and action, and predicts the next frame and reward.
Uses the heuristic dataset instead of random policy dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
file = 'Data/ALE_Breakout-v5_heuristic_transitions.npz'
model_file = 'environment_model_heuristic.pth'
batch_size = 32
num_epochs = 50
learning_rate = 0.001
patience = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the saved frames, actions and rewards as PyTorch tensors


class BreakoutDataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
    def __len__(self):
        return len(self.actions)
    # Scale pixel values to 0-1 and put the channel first for the CNN
    def __getitem__(self, index):
        observation = self.observations[index].astype(np.float32) / 255.0
        next_observation = self.next_observations[index].astype(np.float32) / 255.0
        observation = np.transpose(observation, (2, 0, 1))
        next_observation = np.transpose(next_observation, (2, 0, 1))
        action = np.int64(self.actions[index])
        reward = np.float32(self.rewards[index])
        return (
            torch.from_numpy(observation), torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32), torch.from_numpy(next_observation)
        )


Dataset_file = np.load(file)
print('Total transitions:', len(Dataset_file['actions']))
# Keep the last two of the ten gameplay segments for validation
segment_ends = Dataset_file['segment_end']
segment_end_indices = np.where(segment_ends)[0]
if len(segment_end_indices) < 2:
    raise ValueError('Need more collection segments.')
num_segments = len(segment_end_indices)
num_train_segments = int(num_segments * 0.8)
split_index = segment_end_indices[num_train_segments - 1] + 1


training_data = BreakoutDataset(
    Dataset_file['observations'][:split_index], Dataset_file['actions'][:split_index],
    Dataset_file['rewards'][:split_index], Dataset_file['next_observations'][:split_index]
)
validation_data = BreakoutDataset(
    Dataset_file['observations'][split_index:], Dataset_file['actions'][split_index:],
    Dataset_file['rewards'][split_index:], Dataset_file['next_observations'][split_index:]
)
print('Training / validation transitions:', len(training_data), '/', len(validation_data))
print('\nTraining non-zero rewards:', np.count_nonzero(Dataset_file['rewards'][:split_index]))
print('Validation non-zero rewards:', np.count_nonzero(Dataset_file['rewards'][split_index:]))


training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
# Predicts both the next 84x84 frame and the reward from a frame and action.


class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4, action_embedding_size=16):
        super().__init__()
        # CNN extracts visual features from the current 84x84 frame
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        encoded_size = 64 * 10 * 10
        # Learn features for each of Breakout's four actions
        self.action_embedding = nn.Embedding(num_actions, action_embedding_size)
        # Combine what the screen looks like with the action taken
        self.combined_layer = nn.Sequential(
            nn.Linear(encoded_size + action_embedding_size, 64 * 8 * 8), nn.ReLU()
        )
        # Decoder produces a predicted next frame
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )
        # Resize decoder output from 64x64 back to 84x84
        self.resize = nn.Upsample(size=(84, 84), mode='bilinear', align_corners=False)
        # A separate output predicts the reward
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
        return (predicted_next_observation, predicted_reward)


model = EnvironmentModel().to(device)
observation_loss_function = nn.MSELoss()
reward_loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


training_losses = []
training_observation_losses = []
training_reward_losses = []
validation_losses = []
validation_observation_losses = []
validation_reward_losses = []


best_validation_loss = float('inf')
epochs_without_improvement = 0
best_epoch = 0



# TRAINING AND VALIDATION

for epoch in range(num_epochs):

    # train the model on the training dataset
    model.train()

    total_training_loss = 0.0
    total_training_observation_loss = 0.0
    total_training_reward_loss = 0.0

    for observation, action, reward, next_observation in training_loader:

        # move the data onto the CPU 
        observation = observation.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_observation = next_observation.to(device)

        # matching target shape 4 reward prediction
        reward = reward.unsqueeze(1)

        optimizer.zero_grad()

        # predict the next frame and reward using the current frame and action
        predicted_next_observation, predicted_reward = model(observation, action)

        # calculate how far the predictions are from the actual results
        observation_loss = observation_loss_function(predicted_next_observation, next_observation)
        reward_loss = reward_loss_function(predicted_reward, reward)

        # both losses contribute
        loss = observation_loss + reward_loss

        # backprop to update the model weights
        loss.backward()
        optimizer.step()

        # loss across all batches 
        total_training_loss += loss.item()
        total_training_observation_loss += observation_loss.item()
        total_training_reward_loss += reward_loss.item()

    # averaging loss for whatever epoch it is currently at
    average_training_loss = total_training_loss / len(training_loader)
    average_training_observation_loss = total_training_observation_loss / len(training_loader)
    average_training_reward_loss = total_training_reward_loss / len(training_loader)

    training_losses.append(average_training_loss)
    training_observation_losses.append(average_training_observation_loss)
    training_reward_losses.append(average_training_reward_loss)


    # VALIDATION

    # evaluate on unseen gameplay without updating model weights
    model.eval()

    total_validation_loss = 0.0
    total_validation_observation_loss = 0.0
    total_validation_reward_loss = 0.0

    with torch.no_grad():

        for observation, action, reward, next_observation in validation_loader:

            observation = observation.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_observation = next_observation.to(device)

            reward = reward.unsqueeze(1)

            # predict the next frame and reward on validation data
            predicted_next_observation, predicted_reward = model(observation, action)

            observation_loss = observation_loss_function(predicted_next_observation, next_observation)
            reward_loss = reward_loss_function(predicted_reward, reward)

            loss = observation_loss + reward_loss

            total_validation_loss += loss.item()
            total_validation_observation_loss += observation_loss.item()
            total_validation_reward_loss += reward_loss.item()

    # average validation losses
    average_validation_loss = total_validation_loss / len(validation_loader)
    average_validation_observation_loss = total_validation_observation_loss / len(validation_loader)
    average_validation_reward_loss = total_validation_reward_loss / len(validation_loader)

    validation_losses.append(average_validation_loss)
    validation_observation_losses.append(average_validation_observation_loss)
    validation_reward_losses.append(average_validation_reward_loss)


    # printing results for each epoch, needed as a sanity check + analysis

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train total loss: {average_training_loss:.6f} | Val total loss: {average_validation_loss:.6f}')
    print(f'Train observation loss: {average_training_observation_loss:.6f} | Val observation loss: {average_validation_observation_loss:.6f}')
    print(f'Train reward loss: {average_training_reward_loss:.6f} | Val reward loss: {average_validation_reward_loss:.6f}')


    # EARLY STOPPING

    # as recommended by tutors 
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), model_file)

    else:
        epochs_without_improvement += 1

    # stop training if validation loss hasn't improved for 5 epochs
    if epochs_without_improvement >= patience:
        print('Early stopping')
        break



model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
print('Best epoch:', best_epoch)
print('Best validation loss:', best_validation_loss)
# Plot the total, frame and reward losses separately.
def plot_losses(training, validation, title, ylabel, filename):
    epochs = range(1, len(training) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training, label='Training')
    plt.plot(epochs, validation, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


plot_losses(training_losses, validation_losses,
            'Heuristic Dataset - Total Training and Validation Loss', 'Loss',
            'heuristic_total_loss.png')
plot_losses(training_observation_losses, validation_observation_losses,
            'Heuristic Dataset - Observation Loss', 'Observation MSE',
            'heuristic_observation_loss.png')
plot_losses(training_reward_losses, validation_reward_losses,
            'Heuristic Dataset - Reward Loss', 'Reward MSE',
            'heuristic_reward_loss.png')



model.eval()
number_of_examples = 5
fig, axes = plt.subplots(number_of_examples, 3, figsize=(10, number_of_examples * 3))
with torch.no_grad():
    for i in range(number_of_examples):
        observation, action, reward, next_observation = validation_data[i]
        observation_batch = observation.unsqueeze(0).to(device)
        action_batch = action.unsqueeze(0).to(device)
        predicted_next_observation, predicted_reward = model(observation_batch, action_batch)
        current_frame = observation.squeeze(0).cpu().numpy()
        actual_next_frame = next_observation.squeeze(0).cpu().numpy()
        predicted_frame = predicted_next_observation[0].squeeze(0).cpu().numpy()
        axes[i, 0].imshow(current_frame, cmap='gray')
        axes[i, 0].set_title(f'Current\nAction: {action.item()}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(actual_next_frame, cmap='gray')
        axes[i, 1].set_title(f'Actual next\nReward: {reward.item():.1f}')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(predicted_frame, cmap='gray')
        axes[i, 2].set_title(f'Predicted next\nReward: {predicted_reward.item():.3f}')
        axes[i, 2].axis('off')
plt.tight_layout()
plt.savefig('heuristic_frame_predictions.png')
plt.show()
print('\nTraining finished.')
print('Model saved as:', model_file)
