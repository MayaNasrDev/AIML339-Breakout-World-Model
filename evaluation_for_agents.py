import csv
import gymnasium as gym
import ale_py
import numpy as np

from stable_baselines3 import PPO


environment_name = "ALE/Breakout-v5"

baseline_model = "ppo_baseline"
model_assisted = "ppo_model_ass"

num_episodes = 20
seed = 350

output_file = "ppo_evaluation_results.csv"

gym.register_envs(ale_py)


# same observation setup used for ppo training
class AddChannelDimension(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=env.observation_space.dtype)
        
    def observation(self, observation):
        return observation[:, :, None]
def make_environment():

    env = gym.make(environment_name, obs_type="grayscale", frameskip=1, repeat_action_probability=0.0)

    env = gym.wrappers.ResizeObservation(env, (84, 84))

    env = AddChannelDimension(env)

    return env

# evaluate one of the agents in real breakout
def evaluate_agent(model_file, agent_name):
    model = PPO.load(model_file)
    env = make_environment()
    rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        # same evaluation seeds for both agents
        episode_seed = seed + episode

        observation, info = env.reset(seed=episode_seed)
        observation, reward, terminated, truncated, info = env.step(1)  # FIRE to launch ball because after initial run all results were zero

        terminated = False
        truncated = False

        total_reward = 0
        episode_length = 0
        action_counts = np.zeros(4,dtype=int) #wondering when deterministic what the aciton being chosen most is
        
        while not terminated and not truncated:
            # stochastic the agent is able to explore different moves
            action, _ = model.predict(observation, deterministic=False)
            action_counts[int(action)] +=1

            observation, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            episode_length += 1


        rewards.append(total_reward)
        episode_lengths.append(episode_length)

        print(agent_name, "episode", episode + 1, "reward:", total_reward, "length:", episode_length)
        print("actions chosen:", action_counts)


    env.close()

    return rewards, episode_lengths


print()
print("normal PPO")
print()

baseline_rewards, baseline_lengths = evaluate_agent(baseline_model, "baseline")


print()
print("model assisted PPO")
print()

model_rewards, model_lengths = evaluate_agent(model_assisted, "model_assisted")


# saving each individual episode
with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["agent", "episode", "seed", "reward", "episode_length"])


    for i in range(num_episodes):

        writer.writerow(["baseline", i + 1, seed + i, baseline_rewards[i], baseline_lengths[i]])


    for i in range(num_episodes):

        writer.writerow(["model_assisted", i + 1, seed + i, model_rewards[i], model_lengths[i]])


# summary numbers
baseline_mean = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)

model_mean = np.mean(model_rewards)
model_std = np.std(model_rewards)

# summary results
baseline_mean = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)

model_mean = np.mean(model_rewards)
model_std = np.std(model_rewards)

print("\nFINAL EVALUATION RESULTS")
print("------------------------")

print("\nBaseline PPO")
print(f"Average reward: {baseline_mean:.2f}")
print(f"Standard deviation: {baseline_std:.2f}")
print(f"Lowest reward: {np.min(baseline_rewards):.0f}")
print(f"Highest reward: {np.max(baseline_rewards):.0f}")
print(f"Average episode length: {np.mean(baseline_lengths):.1f}")

print("\nModel-assisted PPO")
print(f"Average reward: {model_mean:.2f}")
print(f"Standard deviation: {model_std:.2f}")
print(f"Lowest reward: {np.min(model_rewards):.0f}")
print(f"Highest reward: {np.max(model_rewards):.0f}")
print(f"Average episode length: {np.mean(model_lengths):.1f}")

print(f"\nDifference in average reward: {model_mean - baseline_mean:+.2f}")
print(f"Results saved to {output_file}")
