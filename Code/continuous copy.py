from util.DataFiles import save
resultsFolder = 'results/continuous/tests/'
def trainAgent(tries, episodes, teacherAgent=None, feedbackProbability = 0, feedbackAccuracy = 0, ppl = False, suffix = 0):
    if teacherAgent == None:
        filenameFolder = resultsFolder + 'rewardsRL'
    else:
        filenameFolder = resultsFolder + 'rewardsIRL'

    for i in range(tries):
        print('Training agent number: ' + str(i+1))

        agent = CartPoleAgentCont()
        rewards = agent.train(episodes, teacherAgent, feedbackProbability, feedbackAccuracy, ppl)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agentRL'+ str(i) + '.npy'
            filenameRewards = filenameFolder + str(i) +'.csv'
        else:
            agentPath = resultsFolder+'/agentIRL'+ str(i) + '_' + str(feedbackProbability) + '_' + str(feedbackAccuracy) + '_' + str(ppl) + str(suffix) + '.npy'
            filenameRewards = filenameFolder + str(i) + '_' + str(feedbackProbability) + '_' + str(feedbackAccuracy) + '_' + str(ppl) + str(suffix) + '.csv'
        agent.save(agentPath)
        save(rewards, filenameRewards)
    return agent
import seaborn as sns
from sklearn.cluster import KMeans
from agent.ContinuousAgent import CartPoleAgentCont
from util.Agent import select_agent
import pickle
import math

agent = CartPoleAgentCont()
teacherAgent, number, teacherPath = select_agent(agent, resultsFolder, 10)
print('Using agent:', number, teacherPath)


with open(resultsFolder + 'agentRL3.npy_gmodelstate', "rb") as f:
    states = pickle.load(f)

arr = []
feedbackProbability = [1, 0.47316, 0.23658]
feedbackAccuracy = [1, 0.9487, 0.47435]

for i in range(2, 12):
    num_cluster = int(math.pow(i, 4))
    print(num_cluster)
    teacherAgent.generalise_model = KMeans(n_clusters=num_cluster, n_init=10)
    teacherAgent.generalise_model.fit(states)
    label = teacherAgent.generalise_model.predict(states)
    teacherAgent.save(resultsFolder + 'agentRL0.npy')

    trainAgent(1, 1000, teacherAgent, feedbackProbability[1], feedbackAccuracy[1], True, num_cluster)