import numpy as np
import pandas as pd
resultsFolder = 'results/tests/'
def get_valueRL(filename) :
    data = []
    for i in range(1):
        fname = resultsFolder + filename + str(i) + '.csv'
        rw = pd.read_csv(fname).iloc[:,1] # read column 1
        rw_refined = get_avg(rw)

        data.append(rw_refined)
    merged = pd.concat(data,axis=1)
    means = np.mean(merged, axis=1)
    return means


def get_value(filename, pro=None, acc=None):
    data = []
    for i in range(1):
        fname = resultsFolder + filename + str(i) + '_' + str(pro) + '_' + str(acc) + '_False' + '.csv'
        rw = pd.read_csv(fname).iloc[:, 1]  # read column 1
        rw_refined = get_avg(rw)
        data.append(rw_refined)
    merged = pd.concat(data, axis=1)

    means = merged.mean(axis=1)
    upper = merged.max(axis=1)
    lower = merged.min(axis=1)
    return means, lower, upper


def get_value_ppr(filename, pro=None, acc=None):
    data = []
    for i in range(3):
        fname = resultsFolder + filename + str(i) + '_' + str(pro) + '_' + str(acc) + '_True' + '.csv'
        rw = pd.read_csv(fname).iloc[:, 1]  # read column 1
        rw_refined = get_avg(rw)
        data.append(rw_refined)

    merged = pd.concat(data, axis=1)

    means = merged.mean(axis=1)
    upper = merged.max(axis=1)
    lower = merged.min(axis=1)
    return means, lower, upper


def get_avg(data):
    # get avg last 200 reward
    num_loop = 100
    rewards = data.tolist()
    avg_reward = [0.] * (len(rewards) - num_loop)
    r_count = 0.
    for i in range(len(rewards)):
        r_count += float(rewards[i])
        if i >= num_loop:
            r_count -= float(rewards[i - num_loop])
            avg_reward[i - num_loop] = r_count / num_loop

    # get avg of cumulative_rewards
    """
    rewards = data.tolist()
    avg_reward = [0.] * len(rewards)
    cumulative_rewards = [0.] * len(rewards)
    cumulated_r = 0.
    for i in range(len(rewards)):
        cumulated_r += rewards[i]
        cumulative_rewards[i] = cumulated_r
    for i in range(len(rewards)):
        avg_reward[i] = cumulative_rewards[i]/ (i+1)
    """

    return pd.Series(avg_reward)
    # return data
RL= get_valueRL("agentRL")
IRLReal, IRLReallower, IRLRealupper = get_value("agentRL", 0.47316, 0.9487)
IRLRealPPR, IRLReallowerPPR, IRLRealupperPPR = get_value_ppr("agentRL", 0.47316, 0.9487)
print(IRLReal[200:250])
print(IRLRealPPR[200:250])