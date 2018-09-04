import time
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv2D, Reshape
from keras.optimizers import Adam
from keras.models import Sequential
import DirectionMap as dir_map
from gym.envs.registration import register


EPISODES = 10000
TEST = False # to evaluate a model
LOAD = False # to load an existing model
MAP_IDS = range(64)
MAP = np.zeros(shape=(8, 8), dtype=np.float32)
MAP[2, 3] = -1.0
MAP[3, 5] = -1.0
MAP[4, 3] = -1.0
MAP[5, 1] = -1.0
MAP[5, 2] = -1.0
MAP[5, 6] = -1.0
MAP[6, 1] = -1.0
MAP[6, 4] = -1.0
MAP[6, 6] = -1.0
MAP[7, 3] = -1.0
MAP[7, 7] = 1.0

WALKABLE = []
for i in range(64):
    row, col = i%8, int(np.floor(i/8))
    state = MAP.copy()
    state[row, col] = 10.0
    WALKABLE += [state]
WALKABLE = np.reshape(WALKABLE, newshape=(64, 8, 8, 1))

# this is Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        if TEST:
            self.epsilon = 0.0
        else:
            self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 500
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()


    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def _build_model(self):
        model = Sequential()

        # input size: [batch_size, 8, 8, 1]
        # output size: [batch_size, 8, 8, 16]
        model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', input_shape=self.state_size, kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 8, 8, 16]
        # output size: [batch_size, 8, 8, 32]
        model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 8, 8, 32]
        # output size: batch_sizex8x8x32 = 2048xbatch_size
        model.add(Reshape(target_shape=(1, 2048)))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def get_qvals(self, state):
        q_value = self.model.predict(state)
        return q_value

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        update_target = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][0, action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][0, action[i]] = reward[i] + self.discount_factor * \
                    target_val[i][0, a]

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # register new env that is not slippery
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '8x8', 'is_slippery': False},
        max_episode_steps=500,
        reward_threshold=0.78, # optimum = .8196
    )

    env = gym.make('FrozenLakeNotSlippery-v0')
    # get size of state and action from environment

    state_size = (8, 8, 1)
    action_size = env.action_space.n

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, filtered_scores = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        row, col = state%state_size[0], int(np.floor(state/state_size[1]))
        state = MAP.copy()
        state[row, col] = 10.0
        state = np.reshape(state, newshape=(1, state_size[0], state_size[1], state_size[2]))

        if LOAD:
            agent.load_model("./FrozenLake_DoubleDQN.h5")

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            next_row, next_col = next_state%state_size[0], int(np.floor(next_state/state_size[0]))
            next_state = MAP.copy()
            next_state[next_row, next_col] = 10.0
            next_state = np.reshape(next_state, newshape=(1, state_size[0], state_size[1], state_size[2]))

            if np.array_equal(state, next_state): # if it tried to go off the map
                reward += -0.1

            if not TEST:
                # save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(state, action, reward, next_state, done)
                # every time step do the training
                agent.train_replay()
            score += reward
            state = next_state

            if done:
                env.reset()
                # every episode update the target model to be same with model
                agent.update_target_model()

                if len(filtered_scores)!=0: # if list is not empty
                    filtered_scores.append(0.98*filtered_scores[-1] + 0.02*score)
                else: # if list is empty
                    filtered_scores.append(score)
                scores.append(score)
                episodes.append(e)
                pylab.gcf().clear()
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig("./FrozenLake_DoubleDQN.png")
                print("episode: {:3}   score: {:8.6}   memory length: {:4}   epsilon {:.3}"
                            .format(e, score, len(agent.memory), agent.epsilon))

                # if the mean of scores of last N episodes is bigger than X
                # stop training
                if np.mean(scores[-min(25, len(scores)):]) >= 0.99:
                    agent.save_model("./FrozenLake_DoubleDQN.h5")

                    qvals = agent.get_qvals(WALKABLE)
                    dir_map.save_map(positions=MAP_IDS, qvalues=qvals, \
                                    map_dim=(state_size[0], state_size[1]), name='FrozenLake_DoubleDQN_' + str(e))
                    time.sleep(1)   # Delays for 1 second
                    sys.exit()

        # save the model every N episodes
        if e % 100 == 0:
            agent.save_model("./FrozenLake_DoubleDQN.h5")

            qvals = agent.get_qvals(WALKABLE)
            dir_map.save_map(positions=MAP_IDS, qvalues=qvals, \
                            map_dim=(state_size[0], state_size[1]), name='FrozenLake_DoubleDQN_' + str(e))

    agent.save_model("./FrozenLake_DoubleDQN.h5")

    qvals = agent.get_qvals(WALKABLE)
    dir_map.save_map(positions=MAP_IDS, qvalues=qvals, \
                    map_dim=(state_size[0], state_size[1]), name='FrozenLake_DoubleDQN_' + str(e))
