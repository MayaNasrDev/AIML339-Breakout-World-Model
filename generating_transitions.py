
"""Collect Breakout transitions using random actions."""

from pathlib import Path
import cv2
import gymnasium as gym
import numpy as np
import ale_py

ENV_NAME = "ALE/Breakout-v5"
Num_Transitions = 10000
Random_Seed = 350
output_file = Path("Data/ALE_Breakout-v5_transitions_test.npz")


def preprocess_frame(frame):
    # convert screenshots to grayscale and resize for the CNN
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized[:, :, np.newaxis].astype(np.uint8)


def collect_transitions(env):
    observations, actions, rewards, next_observations = [], [], [], []
    terminated_values, truncated_values = [], []

    observation, _ = env.reset(seed=Random_Seed)
    env.action_space.seed(Random_Seed)

    action_names = env.unwrapped.get_action_meanings()
    print("Actions:", action_names)

    fire = action_names.index("FIRE") if "FIRE" in action_names else None

    # start the game before collecting transitions
    if fire is not None:
        observation, _, terminated, truncated, _ = env.step(fire)
        if terminated or truncated:
            observation, _ = env.reset()

    print("Collecting random transitions...")

    for i in range(Num_Transitions):
        current_frame = preprocess_frame(observation)

        # original experiment just used random legal actions
        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_frame = preprocess_frame(next_observation)

        # save each state, action, reward and next state
        observations.append(current_frame)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_frame)
        terminated_values.append(terminated)
        truncated_values.append(truncated)

        observation = next_observation

        if (i + 1) % 1000 == 0:
            print(f"Collected {i + 1}/{Num_Transitions} transitions")

        # reset when the game finishes
        if terminated or truncated:
            observation, _ = env.reset()
            if fire is not None:
                observation, _, _, _, _ = env.step(fire)

    return {
        "observations": np.array(observations, dtype=np.uint8),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.uint8),
        "terminated": np.array(terminated_values, dtype=np.bool_),
        "truncated": np.array(truncated_values, dtype=np.bool_)
    }


def main():
    gym.register_envs(ale_py)
    env = gym.make(ENV_NAME, frameskip=1, repeat_action_probability=0.0)

    try:
        data = collect_transitions(env)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **data)

        print("\nSaved to:", output_file)
        print("Observations:", data["observations"].shape)
        print("Non-zero rewards:", np.count_nonzero(data["rewards"]))
        print("Total reward:", np.sum(data["rewards"]))
        print("Natural episode endings:", np.count_nonzero(data["terminated"] | data["truncated"]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
