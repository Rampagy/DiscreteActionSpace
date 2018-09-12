import time
import sys
import pylab
import random
import numpy as np
from GridWorld import Env
from collections import deque
from keras.layers import Dense, Conv2D, Reshape
from keras.optimizers import Adam
from keras.models import Sequential


EPISODES = 5000
TEST = False # to evaluate a model


# this is Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see learning, then change to True
        self.render = False
        self.load = False # load an existing model
        self.save_loc = './GridWorld_DoubleDQN'
        
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99997
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
        
        if self.load:
            self.load_model()


    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def _build_model(self):
        model = Sequential()

        # input size: [batch_size, 10, 10, 1]
        # output size: [batch_size, 10, 10, 16]
        model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', input_shape=self.state_size, kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 5, 5, 16]
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        
        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 10, 10, 32]
        model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 32]
        # output size: batch_sizex10x10x32 = 3200xbatch_size
        model.add(Reshape(target_shape=(1, 3200)))
        model.add(Dense(3000, activation='relu', kernel_initializer='glorot_uniform'))
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
    def load_model(self):
        self.model.load_weights(self.save_loc + '.h5')

    # save the model which is under training
    def save_model(self):
        self.model.save_weights(self.save_loc + '.h5')


if __name__ == "__main__":
    # create environment
    env = Env()
    state = env.reset()

    # get size of state and action from environment
    state_size = (env.observation_size[0], env.observation_size[1], 1)
    action_size = env.action_size # 0 = up, 1 = down, 2 = right, 3 = left

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, filtered_scores = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, (1, state_size[0], state_size[1], state_size[2]))

        if TEST:
            agent.epsilon = 0
            
        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, state_size[0], state_size[1], state_size[2]))

            if not TEST:
                # save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(state, action, reward, next_state, done)
                # every time step do the training
                agent.train_replay()
            score += reward
            state = next_state

            if done:
                env.reset()
                
                if not TEST:
                    # every episode update the target model to be same with model
                    agent.update_target_model()

                if len(filtered_scores) != 0: # if list is not empty
                    filtered_scores.append(0.98*filtered_scores[-1] + 0.02*score)
                else: # if list is empty
                    filtered_scores.append(score)
                scores.append(score)
                episodes.append(e)
                pylab.gcf().clear()
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig(agent.save_loc + '.png')
                print("episode: {:3}   score: {:8.6}   memory length: {:4}   epsilon {:.3}"
                            .format(e, float(score), len(agent.memory), float(agent.epsilon)))

                # if the mean of scores of last N episodes is bigger than X
                # stop training
                if np.mean(scores[-min(25, len(scores)):]) > 0.93:
                    if not TEST:
                        agent.save_model()
                        time.sleep(1)   # Delays for 1 second
                        sys.exit()

        # save the model every N episodes
        if e % 100 == 0:
            if not TEST:
                agent.save_model()

    if not TEST:
        agent.save_model()
