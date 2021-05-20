import gym
import numpy as np
import pandas as pd
import time
import random
import math
from matplotlib import pyplot as plt

from IPython.display import clear_output
from collections import deque

from keras.layers import Dense
from keras.optimizers import Nadam
from keras.models import Sequential



class CartPoleAgentCont:

    def __init__(self, 
            max_steps=100, gamma=0.99, epsilon=1.0, alpha = 0.001, 
            state_shape = None):
        self.env = gym.make("CartPole-v1")

        if(state_shape != None):
            self.state_shape = state_shape
        else: 
            self.state_shape = self.observation_space().shape
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        self.actions_num = self.action_space().n  # 2 actions

        self.envLimits = self.observation_space().high
        self.envLimits[0] = 2.4
        self.envLimits[2] = math.radians(50)

        self.expected_reward = 480
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Interactive
        self.feedbackAmount = 0

        #model & target model
        self.main_network = self.build_network()
        self.target_network = self.build_network()

        #define the update rate at which we want to update the target network
        self.update_rate = 1000 
    
        #copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())
    
    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space

    def build_network(self):
        model = Sequential()
        model.add(Dense(20, input_shape=self.state_shape,  activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.actions_num, activation='linear'))
        opt = Nadam(lr=self.alpha)
        model.compile(loss='mse', optimizer=opt)

        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def test(self, episode = 100):
        rewards = []
        avg_reward = []
        for i in range(episode):
            s = self.env.reset()
            done = False
            reward = 0
            while not done:
                s = self.preprocess_state(s)
                a = self.epsilon_greedy(s)
                s, r, done, _ = self.env.step(a)
                r = self.rewardFunction(s)
                reward += r
            rewards.append(reward)
            avg_reward.append(np.mean(rewards))
        return rewards

    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def epsilon_greedy(self, state, epsilon = 0.1):
        #exploration
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        #exploitation
        else:
            return np.argmax(self.main_network.predict(state)[0])

    def choose_action(self, state, teacherAgent = None, feedbackProbability = 0):
        if (np.random.rand() < feedbackProbability):
            #get advice
            action = np.argmax(teacherAgent.main_network.predict(state)[0])
            self.feedbackAmount += 1
        else:
            action = self.epsilon_greedy(state)
        #endIf
        return action
    

    def preprocess_state(self, state):

        state = np.reshape(state, list((1,) + self.state_shape))

        #crop and resize the image
        # image = state[1:176:2, ::2]

        #convert the image to greyscale
        # image = image.mean(axis=2)

        #improve image contrast
        # image[image==color] = 0

        #normalize the image
        # image = (image - 128) / 128 - 1
        
        #reshape the image
        # image = np.expand_dims(image.reshape(88, 80, 1), axis=0)

        return state

    def updatePolicy(self):
        if len(self.memory) < self.batch_size:
            return # do nothing
        self.trainNetwork(self.batch_size)
        return 
    
    def trainNetwork(self, batch_size):

        #sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.memory, batch_size)
        
        #compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            state = self.preprocess_state(state)
            next_state = self.preprocess_state(next_state)
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward
                
            #compute the Q value using the main network 
            Q_values = self.main_network.predict(state)
            
            Q_values[0][action] = target_Q
            
            #train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)
    def rewardFunction(self, state):
        reward = -1 * abs(state[0])
        if (-self.envLimits[2] < state[2] or state[2] < self.envLimits[2]) or (
            -self.envLimits[0] < state[0] or state[0] < self.envLimits[0] ):
            diff = ( ( ( state[2] / self.envLimits[2] ) **2 ) + ( ( state[0] / self.envLimits[0] ) **2 ) ) / 2
            reward = 1 - diff
        return reward

    def train(self, episodes_num, teacherAgent, feedbackProbability):
        rewards = []
        avg_reward = []
        time_step = 0
        for episode in range(episodes_num):
            if episode > 0 and episode % 10 == 0:
                print(episode)
            state = self.env.reset()
            time_step += 1                
            reward = 0
            for step in range(self.max_steps):
                state = self.preprocess_state(state)
                action = self.choose_action(state, teacherAgent, feedbackProbability)
                next_state, r, done, _ = self.env.step(action)
                
                self.memorize(state, action, reward, next_state, done)
                state = next_state

                # r = self.rewardFunction(state)
                reward += r

                if done:
                    self.update_target_network()
                    break
                self.updatePolicy()



            rewards.append(reward)
            avg_reward.append(np.mean(rewards[-100:]))
        return rewards

    def save(self, filename):
        self.main_network.save_weights(filename)

    def load(self, filename):
        self.main_network.load_weights(filename)
        self.update_target_network()