#Libraries Declaration
import gym
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# sort helping
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def select_agent(agent, agents_folder):
    # MIN_REWARD = 495
    MIN_REWARD = 0.5
    log_file = open(agents_folder + '/log.txt', 'w')

    epochs_num = 500
    dataRL = np.genfromtxt(agents_folder + '/rewardsRL.csv', delimiter=',')

    stdRL = np.std(dataRL, axis=1)

    sorted_std = np.argsort(stdRL)

    agentsWeights = glob.glob(agents_folder + '/agentRL*')
    agentsWeights.sort(key=natural_keys)
    for i in sorted_std:
        print('Checking', agentsWeights[i])
        agent.load(agentsWeights[i])
        rewards = agent.test(epochs_num)
        print(np.mean(rewards))
        if np.mean(rewards) > MIN_REWARD:
            log_file.write('Using agent: ' + str(i) + ' ' + agentsWeights[i])
            log_file.close()
            return agent, i, agentsWeights[i]
    log_file.write('None Found')
    log_file.close()
    return None, None, None
