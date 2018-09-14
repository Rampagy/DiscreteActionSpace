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

EPISODES = 20000
TEST = False

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load = False
        self.save_loc = './GridWorld_A2Critic'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr  = 0.000001
        self.critic_lr = 0.000005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load:
            self.load_model()

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()

        # input size: [batch_size, 10, 10, 1]
        # output size: [batch_size, 10, 10, 16]
        actor.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same', input_shape=self.state_size,
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 10, 10, 32]
        actor.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same',
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 32]
        # output size: batch_sizex10x10x32 = 3200xbatch_size
        actor.add(Reshape(target_shape=(1, 3200)))
        actor.add(Dense(3000, activation='relu',
                    kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                    kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        # input size: [batch_size, 10, 10, 1]
        # output size: [batch_size, 10, 10, 16]
        critic.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same', input_shape=self.state_size,
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 10, 10, 32]
        critic.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same',
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 32]
        # output size: batch_sizex10x10x32 = 3200xbatch_size
        critic.add(Reshape(target_shape=(1, 3200)))
        critic.add(Dense(3000, activation='relu',
                    kernel_initializer='glorot_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                    kernel_initializer='glorot_uniform'))
        critic.summary()
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        advantage = 0

        if done:
            advantage = reward - value
            target[0][0] = reward
        else:
            advantage = reward + self.discount_factor * next_value - value
            target[0][0] = reward + self.discount_factor * next_value

        if advantage < 0:
            # reinforce all actions except the one it took
            advantages += 1
            advantages[0][action] = 0
        else: # (advantage >= 0)
            # only reinforce the action that it took
            advantages[0][action] = 1

        advantages = np.reshape(advantages, (1, advantages.shape[0], advantages.shape[1]))
        target = np.reshape(target, (1, target.shape[0], target.shape[1]))

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    # load the saved model
    def load_model(self):
        self.actor.load_weights(self.save_loc + "_actor.h5")
        self.critic.load_weights(self.save_loc + "_critic.h5")

    # save the model which is under training
    def save_model(self):
        self.actor.save_weights(self.save_loc + "_actor.h5")
        self.critic.save_weights(self.save_loc + "_critic.h5")


if __name__ == "__main__":
    # create environment
    env = Env()
    state = env.reset()

    # get size of state and action from environment
    state_size = (env.observation_size[0], env.observation_size[1], 1)
    action_size = env.action_size # 0 = up, 1 = down, 2 = right, 3 = left

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes, filtered_scores = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, (1, state_size[0], state_size[1], state_size[2]))

        # don't introduce the energy tax complexity right away
        if np.mean(scores[-min(25, len(scores)):]) > 0:
            env.energy_tax = False
        else:
            env.energy_tax = True
        
        while not done:
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, state_size[0], \
                            state_size[1], state_size[2]))

            if not TEST:
                # every time step do the training
                agent.train_model(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if done:
                env.reset()

                if len(filtered_scores) != 0: # if list is not empty
                    filtered_scores.append(0.98*filtered_scores[-1] + 0.02*score)
                else: # if list is empty
                    filtered_scores.append(score)
                scores.append(score)
                episodes.append(e)
                pylab.gcf().clear()
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig(agent.save_loc + '.png')
                print('episode: {:3}   score: {:8.6}   filtered score: {:8.6}'
                        .format(e, float(score), float(np.mean(scores[-min(25, len(scores)):]))))

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
