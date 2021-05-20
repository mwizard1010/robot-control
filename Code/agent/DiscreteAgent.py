import gym
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
# from env.FrozenLakeEnv import FrozenLakeEnv
from IPython.display import clear_output



class FrozenLakeAgent:

    def __init__(self, 
            max_steps=50, learning_rate=0.85,
            gamma=0.95, epsilon=1.0, decay_rate= 0.005):
        self.env = gym.make("FrozenLake-v0", is_slippery=False, map_name="8x8")
        state_num = self.env.observation_space.n  # 8x8 ==> 64
        actions_num = self.env.action_space.n  # four actions

        self.Q = np.zeros((state_num, actions_num))

        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        # Interactive
        self.feedbackAmount = 0

    def test(self, episode = 100):
        rewards = []
        avg_reward = []
        for i in range(episode):
            s = self.env.reset()
            done = False
            reward = 0
            while not done:
                a = self.epsilon_greedy(s)
                s, r, done, _ = self.env.step(a)
                reward += r
            rewards.append(reward)
            avg_reward.append(np.mean(rewards))
        return avg_reward

    def update_Q(self, state, action):
        new_state, reward, done, _ = self.env.step(action)

        target = self.Q[state, action]
        real = reward + self.gamma * np.max(self.Q[new_state])
        error = real - target
        self.Q[state, action] += self.learning_rate * error
        return new_state, reward, done

    def epsilon_greedy(self, state, epsilon = 0.1):
        #exploration
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        #exploitation
        else:
            return np.random.choice(np.where(self.Q[state,:]== np.max(self.Q[state,:]))[0])

    def choose_action(self, state, teacherAgent = None, feedbackProbability = 0):
        if (np.random.rand() < feedbackProbability):
            #get advice
            action = np.argmax(teacherAgent.Q[state])
            self.feedbackAmount += 1
        else:
            action = self.epsilon_greedy(state)
        #endIf
        return action
    


    def decay_epsilon(self, episode):
        self.epsilon = 0.1 + 0.9 * np.exp(-self.decay_rate * episode)

    def train(self, episodes_num, teacherAgent, feedbackProbability):
        rewards = []
        avg_reward = []
        for episode in range(episodes_num):
            state = self.env.reset()
            reward = 0
            for step in range(self.max_steps):
                action = self.choose_action(state, teacherAgent, feedbackProbability)
                state, r, done = self.update_Q(state, action)
                reward += r
                if done:
                    break

            self.decay_epsilon(episode)


            rewards.append(reward)
            avg_reward.append(np.mean(rewards[-100:]))
        return avg_reward

    def save(self, filename):
        np.save(filename, self.Q)

    def load(self, filename):
        self.Q = np.load(filename)
