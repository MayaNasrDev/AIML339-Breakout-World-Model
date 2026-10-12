"""Train a CNN world model on the original random Breakout transitions."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

file = "Data/ALE_Breakout-v5_transitions.npz"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
patience = 5
reward_weight = 10  # extra weight for rare non-zero rewards
model_file = "environment_model_weighted.pth"

#pytorch needs to have the saved transitions in a format where batches can be loaded 
class BreakoutDataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations

    def __len__(self):
        return len(self.actions)
    #tensor conversion, moves channel dim to the front of pytorch
    def __getitem__(self, i):
        # images are saved as 84x84x1 uint8, CNN needs 1x84x84 floats
        obs = torch.tensor(self.observations[i], dtype=torch.float32).permute(2, 0, 1) / 255
        next_obs = torch.tensor(self.next_observations[i], dtype=torch.float32).permute(2, 0, 1) / 255
        action = torch.tensor(self.actions[i], dtype=torch.long)
        reward = torch.tensor(self.rewards[i], dtype=torch.float32)
        return obs, action, reward, next_obs

#uses 80% for training rest for validation
def split_dataset(data):
    # keep complete episodes apart where possible
    split = int(len(data["actions"]) * 0.8) #trying not to split as well in the middle of a game
    ends = data["terminated"] | data["truncated"]
    while split < len(ends) - 1 and not ends[split - 1]:
        split += 1

    def make_dataset(start, stop):
        return BreakoutDataset(data["observations"][start:stop], data["actions"][start:stop], data["rewards"][start:stop],data["next_observations"][start:stop])

    return make_dataset(0, split), make_dataset(split, None)


class EnvironmentModel(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        # current frame -> smaller feature map (image getting smaller, num of feature channels increase)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU()
        )
        # action num -> learned set of features 
        self.action_embedding = nn.Embedding(num_actions, 16)
        #combining img + action features before decoding stage
        self.fc = nn.Sequential(nn.Linear(64 * 8 * 8 + 16, 64 * 8 * 8), nn.ReLU())
        # reconstruct predicted next frame, from learned features 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 6, stride=2), nn.Sigmoid()
        )
        #seperating output in order to predict reward 
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 16, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    
    def forward(self, observation, action):
        visual = self.encoder(observation).view(observation.size(0), -1)
        #joining img + action features
        combined = torch.cat((visual, self.action_embedding(action)), dim=1)
        reward = self.reward_head(combined) 
        features = self.fc(combined).view(-1, 64, 8, 8)
        next_observation = self.decoder(features)
        return next_observation, reward


def plot_losses(train, val, title, filename):
    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="Training loss")
    plt.plot(epochs, val, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def train_model(model, train_loader, val_loader):
    obs_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {key: [] for key in ("train_total", "train_obs", "train_reward",
                                    "val_total", "val_obs", "val_reward")}
    best_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_totals = np.zeros(3)
        for obs, action, reward, next_obs in train_loader:
            optimizer.zero_grad()
            predicted_obs, predicted_reward = model(obs, action)
            obs_loss = obs_loss_fn(predicted_obs, next_obs)
            reward = reward.unsqueeze(1)
            weights = torch.ones_like(reward)
            weights[reward != 0] = reward_weight
            reward_loss = (reward_loss_fn(predicted_reward, reward) * weights).mean()
            loss = obs_loss + reward_loss
            loss.backward()
            optimizer.step()
            train_totals += [loss.item(), obs_loss.item(), reward_loss.item()]

        # validation uses normal, unweighted reward MSE
        model.eval()
        val_totals = np.zeros(3)
        #no updating model weights in order to check performance on data unseen
        with torch.no_grad():
            for obs, action, reward, next_obs in val_loader:
                predicted_obs, predicted_reward = model(obs, action)
                obs_loss = obs_loss_fn(predicted_obs, next_obs)
                reward_loss = reward_loss_fn(predicted_reward, reward.unsqueeze(1)).mean()
                loss = obs_loss + reward_loss
                val_totals += [loss.item(), obs_loss.item(), reward_loss.item()]

        train_totals /= len(train_loader)
        val_totals /= len(val_loader)
        for prefix, values in (("train", train_totals), ("val", val_totals)):
            for name, value in zip(("total", "obs", "reward"), values):
                history[f"{prefix}_{name}"].append(value)

        print(f"Epoch {epoch + 1}/{num_epochs} | train: {train_totals[0]:.6f} | val: {val_totals[0]:.6f}")
        print(f"  Observation: {train_totals[1]:.6f} / {val_totals[1]:.6f}")
        print(f"  Reward: {train_totals[2]:.6f} (weighted) / {val_totals[2]:.6f}")

        if val_totals[0] < best_loss:
            best_loss = val_totals[0]
            no_improvement = 0
            torch.save(model.state_dict(), model_file)
            print("Best model saved")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    return history


def show_predictions(model, val_loader):
    # compare actual and predicted frames from the first validation batch
    obs, action, reward, next_obs = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        predicted_obs, predicted_reward = model(obs, action)

    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    for i in range(5):
        frames = (obs[i, 0], next_obs[i, 0], predicted_obs[i, 0])
        titles = (f"Current frame\nAction: {action[i].item()}", f"Actual next frame\nReward: {reward[i].item():.2f}",f"Predicted next frame\nReward: {predicted_reward[i].item():.2f}")
        for j in range(3):
            axes[i, j].imshow(frames[j].numpy(), cmap="gray")
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig("environment_model_predictions.png")
    plt.show()


def main():
    data = np.load(file)
    train_data, val_data = split_dataset(data)
    print("Training transitions:", len(train_data))
    print("Validation transitions:", len(val_data))
    if not len(val_data):
        raise ValueError("No validation transitions after episode-based split")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model = EnvironmentModel()
    history = train_model(model, train_loader, val_loader)

    plot_losses(history["train_total"], history["val_total"],
                "Environment Model Training and Validation Loss", "environment_model_loss.png")
    plot_losses(history["train_obs"], history["val_obs"],
                "Observation Prediction Loss", "observation_loss.png")
    plot_losses(history["train_reward"], history["val_reward"],
                "Reward Prediction Loss", "reward_loss.png")
    show_predictions(model, val_loader)


if __name__ == "__main__":
    main()
