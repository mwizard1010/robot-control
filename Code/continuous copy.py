"""
@author: Hung Son
"""

import gym

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.policies import CnnPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter

import matplotlib.pyplot as plt
import numpy as np

import os
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("CartPole-v0")
env = Monitor(env, log_dir, allow_early_resets=True)


model = CnnPolicy("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
rewards = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()
    rewards.append(reward)
env.close()
print(log_dir)

results_plotter.plot_results([log_dir], 10000, results_plotter.X_TIMESTEPS, "DDPG LunarLander")

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir, "RL")