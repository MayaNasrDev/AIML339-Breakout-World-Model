**AIML 339 - Model based reinforcement learning in Atari breakout**
Project overview:
The project is an exploration into model based reinforcement learning that uses the Atari game Breakout. The main goal is to investigate if a neural network would be able to learn predictions about how the environment in the game changes after and actions, and if these predictions could help with reinforcement learning. 

*research qs* -> how effectively can a learned environment model predict future observations and support reinforcement learning in a simplified Atari setting? 

CNN was trained in order to predict the next game observation and reward based on the current observation and action. A standard PPO agent was then compared against a model assisted PPO agent that also used added simulated experiences generated from the learned env model. 

- Project completed in Python with Gymnasium, Arcade Learning environment (ALE), PyTorch and Stable-baselines3. 

**My timeline + stages:**
- Firstly, data collection where I collected breakout transition using random actions and a simple heuristic policy. 
- Secondly, I trained a neural network in order to predict the next observation and reward. 
- Then evaluated the prediction errors, reward predictions and short multi step rollouts. 
- Trained the baseline PPO agent with real environment interactions, afterwards trained a model-assisted PPO agent with both real and synthetic experience. 
- Finally compared the trained agents over 20 breeakout episodes using the same evaluation seeds. 

**Results:**
The model assisted agent ended up achieving a very slight improvement when it came to average reward. However, both agent were trained only using a single random seed, meaning these are not sufficient enough results to establish a consistent performance improvement. In the end, the environment models could predict some aspects of the game observations, but moving objects and sparse rewards stayed challenging to predict. 

**Project Files:**
- generating_transitions.py - Collects transitions using random actions.

- generating_transitions_newPolicy.py - Collects transitions using a heuristic policy.

- training_model.py and training_model_heuristic.py - Train the environment models.

- reward_evaluation.py and reward_evaluation_heuristic.py - Evaluate reward predictions.

- rollout_evalutation.py and rollout_evaluation_heuristic.py - Evaluate multi-step predictions.

- learned_breakout_environemnt.py - Implements the learned environment wrapper.

- ppo_baseline.py - Trains the standard PPO agent.

- model_ass_ppo.py - Trains the model-assisted PPO agent.

- evaluation_for_agents.py - Evaluates both PPO agents.

- recording_gameplay.py - Records gameplay using the trained agents.

- results_plots_etc.ipynb - Contains results analysis and visualisations.

Data/ folder -> has the collected transition datasets.
Results/ folder -> contains experiment figures.
Videos/ folder -> gameplay recordings. 

**In terms of reproducibility**:
The environment model checkpoints (.pth) + agents (.zip) are included so that the evaluation scripts get to be run without needing to retrain the models. 

The experiments used random seed 350 for training. 
PPO eval -> 20 episodes with seeds 350 -> 369

ALL REQUIRED PYTHON LIBRARIES IN requirements.txt. 




