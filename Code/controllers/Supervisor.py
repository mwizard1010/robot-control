from controller import Supervisor, Receiver, Emitter, Camera
import pickle
import sys
from collections import deque
import numpy as np
import random
import pandas as pd
from PIL import Image

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


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
        self.batch_size = 64
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay

        self.main_network = self.build_network()
        self.target_network = self.build_network()

        self.pre_state = self.image_process(self.init_image)
        # self.queue_action = []
        self.pre_action = -1

        self.pre_go_straight = False
        self.reward = 0
        self.step = 0
        self.max_step = 200
        self.episode = 0
        self.file = None

        self.finish = False

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
        model.add(Dense(len(self.action_space()), activation='linear'))
        opt = Nadam(learning_rate=self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def save_reward(self, file, rewards):
        rewards_df = pd.DataFrame(rewards)
        rewards_df.to_csv(file)

    def save_model(self, file):
        self.main_network.save_weights(file)

    def load_model(self, file):
        self.main_network.load_weights(file)
        self.update_target()

    def finalise(self, rewards):
        print(self.file + '.csv')
        self.save_reward(self.file + '.csv', rewards)
        self.save_model(self.file + '.model')

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def updatePolicy(self):
        if len(self.memory) < self.batch_size:
            return  # do nothing
        self.trainNetwork(self.batch_size)
        return

    def trainNetwork(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = self.target_network.predict(next_state)
                target_Q = (reward + self.gamma * np.max(target[0]))
            else:
                target_Q = reward
            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)
            # print(Q_values[0], action, target_Q, reward)
            Q_values[0][action] = target_Q
            states.append(state[0])
            targets.append(Q_values[0])
        # train the main network
        states = np.array(states)
        targets = np.array(targets)
        self.main_network.fit(states, targets, epochs=1, verbose=0)

    def normal_action(self, state, epsilon=0.1):
        # exploration
        if np.random.random() <= epsilon:
            action = self.random_action()

        # exploitation
        else:
            action = np.argmax(self.main_network.predict(state))
        return action

    def action_space(self):
        """
        0 : left
        1: right
        2: straight
        """
        return [0, 1, 2]

    def random_action(self):
        return random.choice(self.action_space())

    def propose_action(self, obs):
        return

    def has_obstacle(self, leftValue, rightValue):
        return leftValue > 500 or rightValue > 500

    def back_to_begin(self):
        INITIAL = [0, 0, 0]
        self.trans_field.setSFVec3f(INITIAL)
        ROT_INITIAL = [0, 1, 0, 1.5708]
        self.rot_field.setSFRotation(ROT_INITIAL)

    def reset(self):
        self.step = 0
        self.reward = 0
        self.finish = False
        # self.queue_action.append(-1)
        self.pre_action = -1
        # self.pre_gostraight = False
        self.pre_state = self.image_process(self.init_image)
        self.back_to_begin()
        self.send_to_robot('reset', None)

    def propose_new_action(self, obs):
        left, right = obs
        obstacle_flag = self.has_obstacle(left, right)
        if not obstacle_flag:
            action = 2
            self.pre_go_straight = True
        else:
            # propose new action
            action = self.normal_action(self.pre_state, self.epsilon)
            self.pre_go_straight = False

        self.pre_action = action
        return action

    def execute(self, obs, reward, done, info):

        img = self.get_image()
        state = self.image_process(img)
        if self.pre_action != -1:
            self.reward += reward

            if self.step == self.max_step or done:
                self.memorize(self.pre_state, self.pre_action, reward, state, done)
                self.updatePolicy()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                self.episode += 1
                self.finish = True
                return

            if info:
                self.back_to_begin()

            if not self.pre_go_straight:
                self.memorize(self.pre_state, self.pre_action, reward, state, done)

            if self.episode % 10 == 0 and (self.step + 1) % 100 == 0:
                self.save_image(img, self.episode, self.step)

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
                action = self.propose_new_action(obs)
                if not self.finish:
                    self.send_to_robot('step', action)
                    self.step += 1
            if message == 'reset_done':
                obs = d
                self.execute(obs, 0, False, False)
                action = self.propose_new_action(obs)
                self.send_to_robot('step', action)
                self.step += 1
            if message == 'obstacle':
                self.back_to_begin()

            self.receiver.nextPacket()
        return

    def send_to_robot(self, message, data):
        data = message, data, self.step
        dataSerialized = pickle.dumps(data)
        self.emitter.send(dataSerialized)

    def start(self, max_step, episodes, file):
        self.file = file
        self.max_step = max_step
        rewards = []
        for i in range(episodes):
            self.reset()
            self.episode = i
            while self.supervisor.step(self.timestep) != -1 and not self.finish:
                self.receive_handle()
            print(i, self.reward)
            rewards.append(self.reward)
        self.finalise(rewards)


resultsFolder = 'results/tests/'
supervisor = SupervisorController()
for i in range(1):
    supervisor.start(200, 500, resultsFolder + '/agentRL' + str(i))
