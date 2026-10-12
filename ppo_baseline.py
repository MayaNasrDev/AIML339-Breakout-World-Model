import csv
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

# settings
ENV_NAME = "ALE/Breakout-v5"

NUM_ENVIRONMENTS = 4

# number of real interactions 100000
real_steps = 100000
seed = 350

model_path = "ppo_baseline"
log_file = "ppo_baseline_training.csv"

gym.register_envs(ale_py) # simply registering it with gymnasium


# breakout env using the same atari settings as env used when collecting data
def make_breakout_env(seed):
    def _init():
        env = gym.make(ENV_NAME, obs_type="grayscale", frameskip=1, repeat_action_probability=0.0)

        env = gym.wrappers.ResizeObservation(env, (84, 84)) # resizing to 84*84
        env = AddChannelDimension(env) # adding channel dimension to the observation

        env.reset(seed=seed)

        return env

    return _init


# the channel dimension class
class AddChannelDimension(gym.ObservationWrapper):

    # keeping ppo observations consistent with the learned environment observations
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=env.observation_space.dtype
        )

    # grayscale observation starts as 84*84
    # adding the extra channel so it is the same shape as the environment model data
    def observation(self, observation):
        return observation[:, :, None]


# callback for real step tracking
class RlstepCall_back(BaseCallback):

    # keeping track of real environment interactions within the ppo
    def __init__(self, real_step_budget, log_file):
        super().__init__()

        self.real_step_budget = real_step_budget
        self.log_file = log_file
        self.real_steps = 0


    def _on_training_start(self):

        # start csv for report for graphs etc.
        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ppo_steps", "real_steps"])


    def _on_step(self):

        # ppo steps every environment once each vectorized step
        # there are 4 real environments so this adds 4 each time
        self.real_steps += NUM_ENVIRONMENTS

        # occasionally saving the step counts
        if self.real_steps % 1000 == 0:
            with open(self.log_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.num_timesteps, self.real_steps])

        # stopping when reaching the real interaction budget
        if self.real_steps >= self.real_step_budget:
            return False

        return True


# the four real breakout environments
envs = []

for i in range(NUM_ENVIRONMENTS):
    # slightly different seed for each environment
    envs.append(make_breakout_env(seed + i))


# lets ppo interact with all 4 environments
env = DummyVecEnv(envs)

# recording episode reward + episode length
env = VecMonitor(env)

print(NUM_ENVIRONMENTS, "-> real environments created")


# the ppo agent
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    n_steps=128, # number of steps collected from each env before an update
    batch_size=256,
    gamma=0.99, # discount factor
    verbose=1,
    seed=seed,
    device="auto"
)

print("the ppo model has been created")


# training
callback = RlstepCall_back(real_step_budget=real_steps, log_file=log_file)

print("real environment step budget:", real_steps)

# callback tracks the real interactions
# this will matter more for model assisted ppo since it has real + generated steps
model.learn(total_timesteps=real_steps + NUM_ENVIRONMENTS, callback=callback)


# SAVING THIS TRAINED AGENT
model.save(model_path)

print()
print("the trained agent has been saved to", model_path)
print(callback.real_steps, "real environment interactions have been used for training")
print("saved training log to", log_file)

env.close()