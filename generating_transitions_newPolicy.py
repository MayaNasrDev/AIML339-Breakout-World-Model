"""Collect Breakout transitions using a simple ball-following heuristic."""

from pathlib import Path
import cv2
import gymnasium as gym
import numpy as np
import ale_py

ENV_NAME = "ALE/Breakout-v5"
Num_Transitions = 10000
Segment_Length = 1000
Random_Seed = 350
exploration_rate = 0.10
output_file = Path("Data/ALE_Breakout-v5_heuristic_transitions_test.npz")
rng = np.random.default_rng(Random_Seed)

# screenshots -> grayscale and resizing the cnn
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized[:, :, np.newaxis].astype(np.uint8)


def find_ball_and_paddle(frame):
    # use the pink/red colour of the ball and paddle
    red, green, blue = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    #similar colors for ball and paddle
    mask = (red > 170) & (green > 40) & (green < 110) & (blue > 40) & (blue < 110)
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #separate smaller objects from the paddle
    possible_balls = []
    possible_paddles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 90 < y < 190 and 1 <= w <= 5 and 1 <= h <= 5:
            possible_balls.append((x, y, w, h))
        if y >= 185 and w >= 8:
            possible_paddles.append((x, y, w, h))

    ball_x = None
    paddle_x = None
    #center x pos to decide where to move
    if possible_balls:
        x, y, w, h = possible_balls[0]
        ball_x = x + w / 2
    if possible_paddles:
        x, y, w, h = max(possible_paddles, key=lambda item: item[2])
        paddle_x = x + w / 2

    return ball_x, paddle_x

#starting breakout with same seed 
def collect_transitions(env):
    observations, actions, rewards, next_observations = [], [], [], []
    terminated_values, truncated_values, segment_end_values = [], [], []
    policy_count = 0
    random_count = 0
    ball_count = 0
    paddle_count = 0
    both_count = 0

    observation, _ = env.reset(seed=Random_Seed)
    env.action_space.seed(Random_Seed)
    action_names = env.unwrapped.get_action_meanings()
    print("Actions:", action_names)

    noop = action_names.index("NOOP") if "NOOP" in action_names else None
    fire = action_names.index("FIRE") if "FIRE" in action_names else None
    right = action_names.index("RIGHT") if "RIGHT" in action_names else None
    left = action_names.index("LEFT") if "LEFT" in action_names else None

    # launch the ball before collecting transitions
    if fire is not None:
        observation, _, terminated, truncated, _ = env.step(fire)
        if terminated or truncated:
            observation, _ = env.reset()

    for i in range(Num_Transitions):
        current_frame = preprocess_frame(observation)
        ball_x, paddle_x = find_ball_and_paddle(observation)

        ball_count += ball_x is not None
        paddle_count += paddle_x is not None
        both_count += ball_x is not None and paddle_x is not None

        # random action when detection fails, or for exploration
        use_random = (rng.random() < exploration_rate or
                    ball_x is None or paddle_x is None)

        if use_random:
            action = env.action_space.sample()
            random_count += 1
        else:
            #horizontal pos difference so move towards the ball or stay still if close enough
            difference = ball_x - paddle_x
            if difference < -3 and left is not None:
                action = left
            elif difference > 3 and right is not None:
                action = right
            elif noop is not None:
                action = noop
            else:
                action = env.action_space.sample()
            policy_count += 1

        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_frame = preprocess_frame(next_observation)

        # segment boundaries are separate from actual game endings
        segment_end = ((i + 1) % Segment_Length == 0 or i == Num_Transitions - 1) #reseting every 1000 transitions
        
        #saving curr state, action, reward + next state
        observations.append(current_frame)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_frame)
        terminated_values.append(terminated)
        truncated_values.append(truncated)
        segment_end_values.append(segment_end)

        observation = next_observation

        if (i + 1) % 1000 == 0:
            print(f"Collected {i + 1}/{Num_Transitions} transitions")

        if terminated or truncated or segment_end:
            observation, _ = env.reset()
            if fire is not None:
                observation, _, _, _, _ = env.step(fire)

    print("Heuristic actions:", policy_count)
    print("Random/fallback actions:", random_count)
    print("Ball detected:", ball_count)
    print("Paddle detected:", paddle_count)
    print("Both detected:", both_count)

    return {
        "observations": np.array(observations, dtype=np.uint8),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.uint8),
        "terminated": np.array(terminated_values, dtype=np.bool_),
        "truncated": np.array(truncated_values, dtype=np.bool_),
        "segment_end": np.array(segment_end_values, dtype=np.bool_)
    }


def main():
    gym.register_envs(ale_py)
    env = gym.make(ENV_NAME, frameskip=1, repeat_action_probability=0.0)
    try:
        data = collect_transitions(env)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        #saving everything together so it can be loaded for model training
        np.savez_compressed(output_file, **data)

        print("\nSaved to:", output_file)
        print("Observations:", data["observations"].shape)
        print("Non-zero rewards:", np.count_nonzero(data["rewards"]))
        print("Total reward:", np.sum(data["rewards"]))
        print("Collection segments:", np.count_nonzero(data["segment_end"]))
        print("Natural episode endings:",
            np.count_nonzero(data["terminated"] | data["truncated"]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
