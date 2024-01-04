import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from GameEnvironment import GameEnvironment

class DQNAgentTrainer:
    def __init__(self, env, learning_rate=1e-3, batch_size=32, buffer_size=5000, 
                 learning_starts=10, gamma=0.95, target_update_interval=1000, 
                 train_freq=4, gradient_steps=1, exploration_fraction=0.8, 
                 exploration_initial_eps=1.0, exploration_final_eps=0.01):
        self.env = env
        self.model = DQN(MlpPolicy, self.env, learning_rate=learning_rate, 
                         batch_size=batch_size, buffer_size=buffer_size, 
                         learning_starts=learning_starts, gamma=gamma, 
                         target_update_interval=target_update_interval, 
                         train_freq=train_freq, gradient_steps=gradient_steps, 
                         exploration_fraction=exploration_fraction, 
                         exploration_initial_eps=exploration_initial_eps, 
                         exploration_final_eps=exploration_final_eps)

    def train(self, total_timesteps=5000):
        self.model.learn(total_timesteps=total_timesteps)

    def test(self, episodes=1000):
        self.env.reset_win_counts()  # 勝敗数のリセット
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                self.env.render()


def main(damage_reward_range = 50, front_monster_advantage_reward_range = 10):
    # カスタム環境の作成
    env = GameEnvironment(damage_reward_range, front_monster_advantage_reward_range)

    # DQNAgentTrainerクラスのインスタンス化
    agent_trainer = DQNAgentTrainer(env)

    # 訓練の実行
    agent_trainer.train(total_timesteps=50000)

    # テストの実行
    agent_trainer.test(episodes=1000)
    
    win_rate = env.wiins/1000
    return win_rate

if __name__ == "__main__":
    main()
