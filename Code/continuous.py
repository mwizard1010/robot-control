"""
@author: Hung Son
"""


import os
import gym
from agent.ContinuousAgent import CartPoleAgentCont
from util.DataFiles import save
from util.Agent import select_agent
import random



resultsFolder = 'results/continuous/tests/'
# Create target Directory if don't exist
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)
    print("Directory " , resultsFolder ,  " Created ")


def trainAgent(tries, episodes, teacherAgent=None, feedbackProbability = 0, feedbackAccuracy = 0):
    if teacherAgent == None:
        filenameFolder = resultsFolder + 'rewardsRL'
    else:
        filenameFolder = resultsFolder + 'rewardsIRL'

    for i in range(tries):
        print('Training agent number: ' + str(i+1))

        agent = CartPoleAgentCont()
        rewards = agent.train(episodes, teacherAgent, feedbackProbability, feedbackAccuracy)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agentRL'+ str(i) + '.npy'
            filenameRewards = filenameFolder + str(i) +'.csv'
        else:
            agentPath = resultsFolder+'/agentIRL'+ str(i) + '_' + str(feedbackProbability) + '_' + str(feedbackAccuracy) + '.npy'
            filenameRewards = filenameFolder + str(i) + '_' + str(feedbackProbability) + '_' + str(feedbackAccuracy) +'.csv'
        agent.save(agentPath)
        save(rewards, filenameRewards)
    return agent

def trainAdvisor(agent_num):
    episodes = 700
    trainAgent(agent_num, episodes)
    
if __name__ == "__main__":
    episodes = 1000
    feedbackProbability = [1, 0.47316, 0.23658 ]
    feedbackAccuracy = [1, 0.9487, 0.47435]
    agent_num = 1
    random.seed(0)   
    
    # Interactive RL
    print("Interactive Reinforcement learning ")
    print("Advisor train ... ")
    trainAgent(agent_num, episodes)

    #Sample agent
    agent = CartPoleAgentCont()
    teacherAgent, number, teacherPath = select_agent(agent, resultsFolder, agent_num)
    print('Using agent:', number, teacherPath)

    if(teacherAgent != None):
        # Training with interactive RL
        print('IRL is now training the learner agent with interactive RL')
        for i in range(3):
            learnerAgent = trainAgent(1, episodes, teacherAgent, feedbackProbability[i], feedbackAccuracy[i])

    # print("Finish")

