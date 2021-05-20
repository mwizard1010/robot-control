
import os
import numpy as np
import matplotlib.pyplot as plt

from agent.DiscreteAgent import FrozenLakeAgent
from util.DataFiles import DataFiles
from util.Agent import select_agent

resultsFolder = 'results/discrete/tests/'
# Create target Directory if don't exist
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)
    print("Directory " , resultsFolder ,  " Created ")
else:
    input("Output folder already exist. Press Enter to overwrite...")

files = DataFiles()

def plotRewards(filename):
    dataRL = np.genfromtxt(resultsFolder + filename + 'RL.csv', delimiter=',')
    dataIRL = np.genfromtxt(resultsFolder + filename + 'IRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=0)
    meansIRL = np.mean(dataIRL, axis=0)
    print('meansRL', np.average(meansRL), np.max(meansRL), np.min(meansRL), dataRL.shape)
    print('meansIRL', np.average(meansIRL), np.max(meansIRL), np.min(meansIRL), dataIRL.shape)

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.figure('Collected reward')
    plt.suptitle('Collected reward')

    plt.plot(meansIRL, label = 'Average reward IRL', linestyle = '--', color =  'r')
    plt.plot(meansRL, label = 'Average reward RL', linestyle = '--', color = 'y' )


    plt.legend(loc=4,prop={'size':12})
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.savefig(resultsFolder + filename + '.png')

    plt.show()

#end of plotRewards method

def trainAgent(tries, episodes, teacherAgent=None, feedback=0):
    if teacherAgent == None:
        filenameRewards = resultsFolder + 'rewardsRL.csv'
    else:
        filenameRewards = resultsFolder + 'rewardsIRL.csv'

    files.createFile(filenameRewards)
    for i in range(tries):
        print('Training agent number: ' + str(i+1))



        agent = FrozenLakeAgent(epsilon=0.9, learning_rate=0.01)
        rewards = agent.train(episodes, teacherAgent, feedback)
        suffix = '_i' + str(i) + '_r' + str(rewards)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agentRL'+ str(i) +'.npy'
        else:
            agentPath = resultsFolder+'/agentIRL'+ str(i) +'.npy'

        agent.save(agentPath)
        files.addFloatToFile(filenameRewards, rewards)
    return agent
    
    
if __name__ == "__main__":
    print("Interactive RL for Env is running ... ")
    tries = 2
    episodes = 10000
    feedbackProbability = 0.2

    #Training with discrete RL    
    trainAgent(tries, episodes)

    #sample agent

    agent = FrozenLakeAgent()
    teacherAgent, number, teacherPath = select_agent(agent, resultsFolder)
    print('Using agent:', number, teacherPath)

    if(teacherAgent != None):
        # Training with interactive RL
        print('IRL is now training the learner agent with interactive RL')
        learnerAgent = trainAgent(tries, episodes, teacherAgent, feedbackProbability)

    print("Finish")