from controller import Supervisor, Receiver, Emitter, Camera
import pickle
import sys
from collections import deque
import numpy as np
import random
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from PPR import PPR

# from RobotController import RobotController


class SupervisorController:
    def __init__(self, timesteps=32,
                 gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.99,
                 alpha=0.01
                 ):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("MY_BOT")
        if self.robot_node is None:
            sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
            sys.exit(1)
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        self.timestep = timesteps

        self.camera = Camera('camera')
        self.camera.enable(self.timestep)
        self.init_image = self.get_image()

        self.timestep = timesteps
        self.receiver = Receiver('receiver')
        self.receiver.enable(self.timestep)
        self.emitter = Emitter('emitter')

        self.memory = deque(maxlen=50000)
        self.batch_size = 128
        self.alpha = alpha
        self.gamma = gamma
        self.epsion_init = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay

        self.pre_state = self.init_image
        self.pre_action = -1

        self.pre_go_straight = False
        self.reward = 0
        self.step = 0
        self.max_step = 200
        self.file = None

        # interactive
        self.feedbackProbability = 0
        self.feedbackAccuracy = 1
        self.PPR = False
        self.feedbackTotal = 0
        self.feedbackAmount = 0

        self.init_model()
        self.init_parametter()

    def init_model(self):
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.agent_network = self.build_network()
        self.generalise_model = self.init_gereral_model()
        self.pca_model = self.init_pca_model()

    def init_parametter(self):
        self.epsilon = self.epsion_init
        self.episode = 0
        self.policy_reuse = PPR()

    def init_gereral_model(self):
        n_clusters = 2
        return KMeans(n_clusters=n_clusters, n_init=10)

    def init_pca_model(self):
        n_component = 100
        return PCA(n_components=100, random_state=22)

    def get_image(self):
        image = self.camera.getImage()
        if image is None:
            empty_image = np.zeros((64, 64, 3))
            return Image.fromarray(empty_image.astype(np.uint8))
        else:
            return self.toPIL(image)

    def toPIL(self, bytes_data):
        imgPIL = Image.frombytes('RGBA', (64, 64), bytes_data)
        imgPIL = imgPIL.convert('RGB')
        return imgPIL

    def image_process(self, PIL):
        array = np.array(PIL)
        array = array / 255
        return np.reshape(array, list((1,) + array.shape))

    def save_image(self, PIL, ep, step):
        PIL.save(resultsFolder + 'images/' + str(ep) + '_' + str(step) + '.png')

    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def observation_space(self):
        return self.observation_space

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def build_network(self):
        model = Sequential()
        model.add(Input(shape=(64, 64, 3)))
        model.add(Conv2D(4, kernel_size=8, activation='linear', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, kernel_size=4, activation='linear', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(16, kernel_size=2, activation='linear', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='linear'))
        model.add(Dense(len(self.action_space()), activation='softmax'))
        opt = Nadam(learning_rate=self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def save_reward(self, file, rewards, totals, feedbacks):
        pairs = {'Reward': rewards, 'Total': totals, 'Feedback': feedbacks}
        data_df = pd.DataFrame.from_dict(pairs)
        data_df.to_csv(file)

    def save_model(self, file):
        self.main_network.save_weights(file)
        self.save_generalise_model(file)

    def save_generalise_model(self, filename):
        obs = [s[5] for s in self.memory]
        with open(filename + 'gel', "wb") as f:
            pickle.dump(self.generalise_model, f)
        with open(filename + 'pca', "wb") as f:
            pickle.dump(self.pca_model, f)
        with open(filename + 'state', "wb") as f_:
            pickle.dump(obs, f_)

    def load_generalise_model(self, filename):
        with open(filename + 'gel', "rb") as f:
            print(filename + 'gel')
            self.generalise_model = pickle.load(f)
        with open(filename + 'pca', "rb") as f:
            self.pca_model = pickle.load(f)

    def load_model(self, file):
        self.agent_network.load_weights(file + '.model')
        self.load_generalise_model(file + '.model')
        self.update_target()

    def finalise(self, rewards, totals, feedbacks, ppr):
        file = self.file + '_' + str(self.feedbackProbability) + '_' + str(self.feedbackAccuracy) + str(ppr)
        self.save_reward(file + '.csv', rewards, totals, feedbacks)
        self.save_model(file + '.model')

    def get_group(self, state):
        # nx, ny, nz = state[0].shape
        # state = state.reshape(nx * ny * nz)
        # state = [state]
        # new_state = self.pca_model.transform(state)

        nx, ny, nz = state[0].shape
        image_grayscale = state[0].mean(axis=2).astype(np.float32)
        image_grayscale = image_grayscale.reshape(nx * ny)
        image_grayscale = [image_grayscale]
        return self.generalise_model.predict(image_grayscale)[0]

    def memorize(self, state, action, reward, next_state, done, obs):
        self.memory.append((state, action, reward, next_state, done, obs))

    def updatePolicy(self, batchSize = 0):
        if batchSize == 0:
            batchSize = self.batch_size
        if len(self.memory) < batchSize:
            self.trainNetwork(len(self.memory))
            return  # do nothing
        self.trainNetwork(batchSize)
        return

    def trainNetwork(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done, obs in minibatch:
            state_processed = self.image_process(state)
            next_state_processed = self.image_process(next_state)

            if not done:
                target = self.target_network.predict(next_state_processed)
                target_Q = (reward + self.gamma * np.max(target[0]))
            else:
                target_Q = reward
            # compute the Q value using the main network
            Q_values = self.main_network.predict(state_processed)
            Q_values[0][action] = target_Q
            states.append(state_processed[0])
            targets.append(Q_values[0])
        # train the main network
        states = np.array(states)
        targets = np.array(targets)
        self.main_network.fit(states, targets, epochs=1, verbose=0)

    def normal_action(self, state, epsilon=0.1):
        # exploration
        if np.random.random() <= epsilon:
            action = self.random_action()
            # PPR:
            if self.PPR:
                group = self.get_group(state)
                redoAction, rate = self.policy_reuse.get(group)
                # print(group, rate)
                if (np.random.rand() < rate):
                    action = redoAction
            # end PPR:

        # exploitation
        else:
            action = np.argmax(self.main_network.predict(state))
        return action

    def action_space(self):
        """
        0: left
        1: right
        2: straight
        """
        return [0, 1, 2]

    def random_action(self):
        # if np.random.rand() < 0.5:
        #     return 2
        # else:
        #     return random.choice([0, 1])
        return random.choice(self.action_space())

    def propose_action(self, obs):
        return

    def has_obstacle(self, leftValue, rightValue):
        return leftValue > 500 or rightValue > 500

    def back_to_begin(self):
        INITIAL = [0, 0, 0]
        self.trans_field.setSFVec3f(INITIAL)
        ROT_INITIAL = [0, 1, 0, 3.2]
        self.rot_field.setSFRotation(ROT_INITIAL)

    def reset(self):
        self.pre_state = self.init_image
        self.pre_action = -1

        self.pre_go_straight = False
        self.reward = 0
        self.step = 0
        self.finish = False

        self.feedbackTotal = 0
        self.feedbackAmount = 0

        self.back_to_begin()
        self.send_to_robot('reset', None)

    def propose_new_action(self, obs):
        left, right = obs
        obstacle_flag = self.has_obstacle(left, right)

        pre_state_processed = self.image_process(self.pre_state)
        if not obstacle_flag:
            action = 2
            self.pre_action = action
            self.pre_go_straight = True
        else:

            # propose new action ------------------
            if self.PPR:
                self.policy_reuse.step()
            if np.random.rand() < self.feedbackProbability:
                # get advice
                trueAction = np.argmax(self.agent_network.predict(pre_state_processed))

                # PPR:
                if self.PPR:
                    group = self.get_group(pre_state_processed)
                    self.policy_reuse.add(group, trueAction)
                # end PPR:

                if np.random.rand() < self.feedbackAccuracy:
                    action = trueAction
                else:
                    while True:
                        action = self.random_action()
                        if action != trueAction:
                            break
                self.feedbackAmount += 1
            else:
                action = self.normal_action(pre_state_processed, self.epsilon)
            self.pre_go_straight = False
            self.feedbackTotal += 1

            self.pre_action = action
        return action

    def execute(self, obs, reward, done, info):


        state = self.get_image()
        if self.pre_action != -1:
            self.reward += reward
            if self.step == self.max_step or done:
                if done:
                    self.save_image(state, self.episode, self.step)
                    self.save_image(self.pre_state, self.episode, self.step - 1)

                self.memorize(self.pre_state, self.pre_action, reward, state, done, obs)
                self.updatePolicy(self.step)
                self.update_target()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                self.episode += 1
                self.finish = True
                return

            if info:
                self.back_to_begin()

            if not self.pre_go_straight:
                self.memorize(self.pre_state, self.pre_action, reward, state, done, obs)
                self.pre_state = state

        return

    def receive_handle(self):
        send_message, send_data = None, None
        if self.receiver.getQueueLength() > 0:
            data = self.receiver.getData()
            message, d = pickle.loads(data)
            if message == 'step_done':
                obs, r, d, i, s = d
                # print(s, self.step - 1, self.pre_action, r)  # check synchronize

                self.execute(obs, r, d, i)
                if not self.finish:
                    action = self.propose_new_action(obs)
                    self.send_to_robot('step', action)
                    self.step += 1
            if message == 'reset_done':
                obs = d
                self.execute(obs, 0, False, False)
                action = self.propose_new_action(obs)
                self.send_to_robot('step', action)
            if message == 'obstacle':
                self.back_to_begin()

            self.receiver.nextPacket()
        return

    def send_to_robot(self, message, data):
        data = message, data, self.step
        dataSerialized = pickle.dumps(data)
        self.emitter.send(dataSerialized)

    def start(self, max_step, episodes, file,
              feedbackP=0, feedbackA=1, PPR=False):
        self.file = file
        self.max_step = max_step
        self.feedbackProbability = feedbackP
        self.feedbackAccuracy = feedbackA
        self.PPR = PPR
        rewards = []
        feedbackTotal = []
        feedbackAmount = []
        self.init_parametter()
        for i in range(episodes):
            self.reset()
            self.episode = i
            while self.supervisor.step(self.timestep) != -1 and not self.finish:
                self.receive_handle()
            print(i, self.reward, self.feedbackTotal, self.feedbackAmount)
            rewards.append(self.reward)
            feedbackTotal.append(self.feedbackTotal)
            feedbackAmount.append(self.feedbackAmount)
        self.finalise(rewards, feedbackTotal, feedbackAmount, PPR)


resultsFolder = 'results/tests/'
supervisor = SupervisorController()
feedbackProbability = [1, 0.47316, 0.23658, 0]
feedbackAccuracy = [1, 0.9487, 0.47435, 0]


supervisor.start(300, 500, resultsFolder + 'agentRL' + str(0), feedbackProbability[3], feedbackAccuracy[3], False)
# supervisor.load_model('results/tests/agentRL4')
# supervisor.start(300, 500, resultsFolder + 'agentRL' + str(0), feedbackProbability[1], feedbackAccuracy[1], False)
for i in range(1):
    if not PPR:
        supervisor.init_model()
    else:
        supervisor.load_model('results/tests/agentRL4')
    supervisor.start(300, 500, resultsFolder + 'agentRL' + str(i), feedbackProbability[1], feedbackAccuracy[1], False)

# with open(resultsFolder + 'agentRL0_0_0_True.modelstate', "rb") as f:
#     states = pickle.load(f)
#
# reshape_states = []
# for s in states:
#     nx, ny, nz = s.shape
#     new_state = s.reshape(nx * ny * nz)
#     reshape_states.append(new_state)
#
# pca = PCA(n_components=100, random_state=22)
# reshape_states = pca.fit_transform(reshape_states)
#
# print(reshape_states[0].shape)
#
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(reshape_states)
#
# supervisor.generalise_model = kmeans
# supervisor.pca_model = pca
#
# supervisor.save_generalise_model(resultsFolder + 'agentRL0.model')
