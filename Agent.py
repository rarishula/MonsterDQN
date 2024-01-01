import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from GameEnvironment import GameEnvironment

# Create custom environment
env = GameEnvironment()

# Construct the DQN agent
model = DQN(MlpPolicy, env, learning_rate=1e-5, batch_size=32, buffer_size=50000, 
            learning_starts=10, gamma=0.95, target_update_interval=1000, 
            train_freq=4, gradient_steps=1, exploration_fraction=0.8, 
            exploration_initial_eps=1.0, exploration_final_eps=0.01)
            
# Execute training
model.learn(total_timesteps=50000)

# Execute testing
episodes = 1000
env.reset_win_counts()  # 勝敗数のリセット
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
