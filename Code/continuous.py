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


def trainAgent(tries, episodes, teacherAgent=None, feedback=0):
    if teacherAgent == None:
        filenameRewards = resultsFolder + 'rewardsRL'
    else:
        filenameRewards = resultsFolder + 'rewardsIRL'

    for i in range(tries):
        print('Training agent number: ' + str(i+1))

        agent = CartPoleAgentCont()
        rewards = agent.train(episodes, teacherAgent, feedback)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agentRL'+ str(i) +'.npy'
        else:
            agentPath = resultsFolder+'/agentIRL'+ str(i) +'.npy'

        agent.save(agentPath)
        save(rewards, filenameRewards + str(i) +'.csv')
    return agent

def trainAdvisor(agent_num):
    episodes = 300
    trainAgent(agent_num, episodes)
    
if __name__ == "__main__":
    episodes = 300
    feedbackProbability = 0.3
    agent_num = 3
    random.seed(0)

    #Reinforcement learning
    # print("RL")
    # trainAgent(1, episodes)


    #interactive RL
    # print("Advisor train ... ")
    # trainAdvisor(agent_num)

    #sample agent
    agent = CartPoleAgentCont()
    teacherAgent, number, teacherPath = select_agent(agent, resultsFolder, agent_num)
    print('Using agent:', number, teacherPath)

    if(teacherAgent != None):
        # Training with interactive RL
        print('IRL is now training the learner agent with interactive RL')
        learnerAgent = trainAgent(3, episodes, teacherAgent, feedbackProbability)

    # print("Finish")

