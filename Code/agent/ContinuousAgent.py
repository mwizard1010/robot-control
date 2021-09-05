import gym
import numpy as np
import pandas as pd
import time
import random
import math
import pickle
from collections import deque

from tensorflow.keras import models
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential

from agent.PPR import PPR



class CartPoleAgentCont:

    def __init__(self,
            max_steps=200, reward_threshold = 195,
            gamma=0.99, 
            epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, 
            alpha = 0.01):
        self.env = gym.make("CartPole-v0")

        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay

        # Interactive
        self.feedbackAmount = 0

        #model 
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.generalise_model = self.init_gereral_model()
        self.update_target()

        self.policy_reuse = PPR()

    def init_gereral_model(self):
        n_clusters = 3
        return KMeans(n_clusters=n_clusters, n_init=10)
    
    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def build_network(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.observation_space().shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space().n, activation='linear'))
        opt = Nadam(learning_rate=self.alpha)
        model.compile(loss='mse', optimizer=opt)

        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def normal_action(self, state, epsilon = 0.1):
        #exploration
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        #exploitation
        else:
            return np.argmax(self.main_network.predict(state))

    def get_group(self, state, teacherAgent):
        
        # inp = self.main_network.input
        # out = self.main_network.layers[0].output

        # tmp_model = models.Model(inp, out)
        # return np.argmax(tmp_model.predict(input)[0])
        return teacherAgent.generalise_model.predict(state)[0]


    def choose_action(self, state, epsilon, teacherAgent = None, feedbackProbability = 0, feedbackAccuracy = 0, ppl = False):
        
        if ppl == True:
            self.policy_reuse.step()


        if (np.random.rand() < feedbackProbability):
            #get advice
            trueAction = np.argmax(teacherAgent.main_network.predict(state)[0])

            # PPR: 
            if ppl == True:
                group = self.get_group(state, teacherAgent)
                self.policy_reuse.add(group, trueAction)
            # end PPR:

            if (np.random.rand() < feedbackAccuracy):
                action = trueAction
            else:
                while True: 
                    action = self.env.action_space.sample()
                    if  action != trueAction:
                        break
            self.feedbackAmount += 1
        else:
            action = self.normal_action(state, epsilon)     
            #PPR:  
            if ppl == True:
                group = self.get_group(state, teacherAgent)
                redoAction, rate = self.policy_reuse.get(group)
                # print(redoAction, rate)
                if (np.random.rand() < rate):
                    action = redoAction
                    self.feedbackAmount += 1
            #end PPR:
        return action
    

    def preprocess_state(self, state):
        state = np.reshape(state, list((1,) + self.observation_space().shape))

        return state

    def updatePolicy(self):
        if len(self.memory) < self.batch_size:
            return # do nothing
        self.trainNetwork(self.batch_size)
        return 
    
    def train_generalise_model(self):
        states = [s[0][0] for s in self.memory]
        self.generalise_model.fit(states)

    def trainNetwork(self, batch_size):

        #sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = self.target_network.predict(next_state)
                target_Q = (reward + self.gamma * np.max(target[0]))
            else:
                target_Q = reward
            #compute the Q value using the main network 
            Q_values = self.main_network.predict(state)
            Q_values[0][action] = target_Q
            states.append(state[0])
            targets.append(Q_values[0])
        #train the main network
        states = np.array(states)
        targets = np.array(targets)
        #print(states, targets)
        self.main_network.fit(states, targets, epochs=1, verbose=0)

    def train(self, episodes_num, teacherAgent, feedbackProbability, feedbackAccuracy, ppl):
        rewards = []
        time_step = 0

        # rerun 5 step at first without recording
        max = episodes_num + 5
        for episode in range(max):
            if episode > 0 and episode % 50 == 0:
                print(episode)
            state = self.env.reset()
            state = self.preprocess_state(state)
            time_step += 1                
            reward = 0
            for step in range(self.max_steps):
                action = self.choose_action(state, self.get_epsilon(episode), teacherAgent, feedbackProbability, feedbackAccuracy, ppl)
                next_state, r, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                
                reward += r
                if done:
                    if (reward < self.max_steps):
                        # punish for done
                        self.memorize(state, action, -1, next_state, done)
                    else :
                        self.memorize(state, action, 2, next_state, done)
                    self.update_target()
                    break

                self.memorize(state, action, r, next_state, done)
                state = next_state
                
            if(episode >= 5):
                rewards.append(reward)  
                print(reward, self.feedbackAmount)
            # if (np.mean(rewards) < 190):
            self.updatePolicy()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.feedbackAmount = 0
  

            # avg_reward.append(np.mean(rewards[-100:]))
        return rewards

    def save(self, filename):
        self.main_network.save_weights(filename)
        self.save_generalise_model(filename + '_gmodel')

    def load(self, filename):
        self.main_network.load_weights(filename)
        self.load_generalise_model(filename + '_gmodel')
        self.update_target()

    def save_generalise_model(self, filename):
        # states = self.train_generalise_model()
        states = [s[0][0] for s in self.memory]
        with open(filename, "wb") as f:
            pickle.dump(self.generalise_model, f)
        with open(filename + 'state', "wb") as f_:
            pickle.dump(states, f_)
        
    def load_generalise_model(self, filename):
        with open(filename, "rb") as f:
            self.generalise_model = pickle.load(f)

    def test(self, episode = 50):
        rewards = []
        avg_reward = []
        for i in range(episode):
            s = self.env.reset()
            done = False
            reward = 0
            while not done:
                s = self.preprocess_state(s)
                a = self.normal_action(s)
                s, r, done, _ = self.env.step(a)
                reward += r
            rewards.append(reward)
            avg_reward.append(np.mean(rewards))
        return rewards