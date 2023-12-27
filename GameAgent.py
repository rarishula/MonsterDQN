import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from GameEnvironment import GameEnvironment


# カスタム環境の作成
env = GameEnvironment()


# ニューラルネットワークモデルの構築
model = Sequential()
model.add(Dense(24, activation='relu', input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# DQNエージェントの構築
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# 訓練の実行
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# テストの実行
dqn.test(env, nb_episodes=5, visualize=False)
