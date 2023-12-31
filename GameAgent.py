import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from GameEnvironment import GameEnvironment


# custom environment
env = GameEnvironment()


# built model
model = Sequential()
model.add(Dense(24, activation='relu', input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# built DQN agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# do fit
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# do test
dqn.test(env, nb_episodes=5, visualize=False)
