"""
@author: Hung Son
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from agent.ContinuousAgent import CartPoleAgentCont
from util.DataFiles import DataFiles
from util.Agent import select_agent

resultsFolder = 'results/continuos/tests/'
# Create target Directory if don't exist
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)
    print("Directory " , resultsFolder ,  " Created ")
else:
    input("Output folder already exist. Press Enter to overwrite...")

files = DataFiles()

def trainAgent(tries, episodes, teacherAgent=None, feedback=0):
    if teacherAgent == None:
        filenameRewards = resultsFolder + 'rewardsRL.csv'
    else:
        filenameRewards = resultsFolder + 'rewardsIRL.csv'

    files.createFile(filenameRewards)
    for i in range(tries):
        print('Training agent number: ' + str(i+1))



        agent = CartPoleAgentCont()
        rewards = agent.train(episodes, teacherAgent, feedback)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agentRL'+ str(i) +'.npy'
        else:
            agentPath = resultsFolder+'/agentIRL'+ str(i) +'.npy'

        agent.save(agentPath)
        files.addFloatToFile(filenameRewards, rewards)
    return agent
    
    
if __name__ == "__main__":
    agent = CartPoleAgentCont()
    print(agent.state_shape)

    # print("Interactive RL for Env is running ... ")
    # tries = 2
    # episodes = 100
    # feedbackProbability = 0.3

    # # play(agent, num_episodes=5)

    # #Training with autonomous RL    
    # trainAgent(tries, episodes)

    # #sample agent

    # agent = CartPoleAgentCont()
    # teacherAgent, number, teacherPath = select_agent(agent, resultsFolder)
    # print('Using agent:', number, teacherPath)

    # if(teacherAgent != None):
    #     # Training with interactive RL
    #     print('IRL is now training the learner agent with interactive RL')
    #     learnerAgent = trainAgent(tries, episodes, teacherAgent, feedbackProbability)

    # print("Finish")

