from controller import Robot, Emitter, Camera
import random
import numpy as np
import math
from collections import deque
from PIL import Image

from PPR import PPR
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os


class RobotController:
    def __init__(self, max_steps=2, init_position=(0, 0, 0), final_position=(-0.3, 0, 0.3), max_speed=2,
                 gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995,
                 alpha=0.01
                 ):
        self.robot = Robot()

        self.time_step = int(self.robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.max_speed = max_speed

        self.setupDevice()

        self.init_position = init_position
        self.current_position = init_position
        self.final_position = final_position
        self.num_collision = 0

        self.done = False

        self.emitter = Emitter('emitter')

        self.memory = deque(maxlen=50000)
        self.batch_size = 16
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay

        # Interactive
        self.feedbackAmount = 0

        # model
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.generalise_model = self.init_gereral_model()
        self.update_target()

        self.policy_reuse = PPR()

    def init_gereral_model(self):
        n_clusters = 3
        return KMeans(n_clusters=n_clusters, n_init=10)

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

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def normal_action(self, state, epsilon=0.1):
        # exploration
        if np.random.random() <= epsilon:
            action = self.random_action()
        # exploitation
        else:
            action = np.argmax(self.main_network.predict(state))
        return action

    def setupDevice(self):
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        self.rightDistanceSensor = self.robot.getDevice('ds1')
        self.leftDistanceSensor = self.robot.getDevice('ds0')
        self.rightDistanceSensor.enable(self.time_step)
        self.leftDistanceSensor.enable(self.time_step)

        self.gps = self.robot.getDevice('gps')
        self.touchSensor1 = self.robot.getDevice('touch_sensor1')
        self.touchSensor2 = self.robot.getDevice('touch_sensor2')
        self.touchSensor3 = self.robot.getDevice('touch_sensor3')
        self.touchSensor4 = self.robot.getDevice('touch_sensor4')
        self.touchSensor5 = self.robot.getDevice('touch_sensor5')
        self.gps.enable(self.time_step)
        self.touchSensor1.enable(self.time_step)
        self.touchSensor2.enable(self.time_step)
        self.touchSensor3.enable(self.time_step)
        self.touchSensor4.enable(self.time_step)
        self.touchSensor5.enable(self.time_step)

        self.camera = Camera('camera')
        self.camera.enable(self.time_step)

        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)

        self.init_leftValue = self.leftDistanceSensor.getValue()
        self.init_rightValue = self.rightDistanceSensor.getValue()
        self.init_image = self.get_image()

    def is_collised(self):
        if (self.touchSensor1.getValue() + self.touchSensor2.getValue() +
                self.touchSensor3.getValue() + self.touchSensor4.getValue() +
                self.touchSensor5.getValue()) > 0:
            return True
        gpsValue = self.gps.getValues()
        self.current_position = gpsValue
        if (self.current_position[0] < - 0.5 or self.current_position[0] > 0.5 or
            self.current_position[2] < - 0.5 or self.current_position[2] > 0.5 ):
            return True


        return False

    def step(self, a):
        if not self.done:
            self.robot.step(timestep)
            if not self.is_collised():
                leftValue = self.leftDistanceSensor.getValue()
                rightValue = self.rightDistanceSensor.getValue()
                reward = 0
                if a == 0:
                    leftSpeed, rightSpeed = self.turnLeft(leftValue, rightValue)
                elif a == 1:
                    leftSpeed, rightSpeed = self.turnRight(leftValue, rightValue)
                elif a == 2:
                    leftSpeed, rightSpeed = self.goStraight()
                    reward += leftSpeed
                # elif a == 3:
                #     leftSpeed, rightSpeed = self.goSlow()

                self.leftMotor.setVelocity(leftSpeed)
                self.rightMotor.setVelocity(rightSpeed)

                # set observation .............

                observations = self.get_image(), leftValue, rightValue

                # set reward .............

                # set done .........
                self.set_done()
                return observations, reward, self.done, False

            else:
                self.num_collision += 1
                observations = self.reset()
                self.
                reward = -100
                return observations, reward, False, True

        return self.get_image(), None, self.done

    def get_image(self):
        image = self.camera.getImage()
        if image is None:
            return np.zeros((64, 64, 3))
        else:
            return self.toPIL(image)

    def has_obstacle(self, leftValue, rightValue):
        return leftValue > 500 or rightValue > 500

    def speed_at_obstacle(self, leftValue, rightValue):
        leftSpeed, rightSpeed = 0, 0
        if leftValue > 500:
            if rightValue > 500:
                leftSpeed, rightSpeed = self.moveBack()
            else:
                leftSpeed, rightSpeed = self.turnRight(leftValue, rightValue)
        elif rightValue > 500:
            leftSpeed, rightSpeed = self.turnLeft(leftValue, rightValue)
        else:
            leftSpeed, rightSpeed = self.goStraight()
        return leftSpeed, rightSpeed

    def set_done(self):
        gpsValue = self.gps.getValues()
        self.current_position = gpsValue
        if abs(self.current_position[0] - self.final_position[0]) <= 0.08 and \
                abs(self.current_position[2] - self.final_position[2]) <= 0.08:
            self.done = True

    def action_space(self):
        """
        0 : left
        1: right
        2: straight
        3: slow
        """
        return [0, 1, 2]

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
            Q_values[0][action] = target_Q
            states.append(state[0])
            targets.append(Q_values[0])
        # train the main network
        states = np.array(states)
        targets = np.array(targets)
        # print(states, targets)
        self.main_network.fit(states, targets, epochs=1, verbose=0)

    def random_action(self):
        # if not self.is_collised():
        #     # return random.choice([0, 1, 2, 3])
        #     if np.random.rand() < 0.5:
        #         return 0
        #     else:
        #         return random.choice([1, 2, 3])
        # else:
        #     print('Collision!')
        #     return 1
        return random.choice(self.action_space())

    def goStraight(self):
        return self.max_speed, self.max_speed

    def goSlow(self):
        return self.max_speed / 4, self.max_speed / 4

    def turnLeft(self, leftDistance, rightDistance):
        return -(leftDistance / 100), (rightDistance / 100) + 0.5

    def turnRight(self, leftDistance, rightDistance):
        return (leftDistance / 100) + 0.5, -(rightDistance / 100)

    def reset(self):
        self.emitter.send('Reset'.encode())
        self.done = False
        return self.init_image, self.init_leftValue, self.init_rightValue

    def toPIL(self, bytes):
        imgPIL = Image.frombytes('RGBA', (64, 64), bytes)
        imgPIL = imgPIL.convert('RGB')
        return imgPIL

    def image_process(self, PIL):
        array = np.array(PIL)
        return np.reshape(array, list((1,) + array.shape))

    def save_image(self, PIL, ep, step):
        PIL.save(resultsFolder + 'images/' + str(ep) + '_' + str(step) + '.png')

    def train(self, episodes):
        rewards = []
        for ep in range(episodes):
            reward = 0
            img, left, right = self.reset()
            state = self.image_process(img)
            for step in range(self.max_steps):
                obstac_flag = self.has_obstacle(left, right)
                if obstac_flag:
                    action = self.normal_action(state)
                else:
                    action = 2  # go straight
                obs, r, d, i = self.step(action)
                img, left, right = obs
                next_state = self.image_process(img)

                if d:
                    reward += 1000
                    print(step, ':', reward)
                    self.memorize(state, action, reward, next_state, d)
                    self.update_target()
                    break
                if obstac_flag or i:
                    self.memorize(state, action, r, next_state, d)
                state = next_state
                reward += r

                if ep % 5 == 0 and step % 100 == 0:
                    self.update_target()
                    if obs is not None:
                        self.save_image(img, ep, step)
                if step == self.max_steps - 1:
                    print('Not done:', reward)

            rewards.append(reward)
            self.updatePolicy()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        print('num_collision:', self.num_collision)
        return rewards

    def save(self, filename):
        self.main_network.save_weights(filename)
        # self.save_generalise_model(filename + '_gmodel')

    def load(self, filename):
        self.main_network.load_weights(filename)
        # self.load_generalise_model(filename + '_gmodel')
        self.update_target()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd


def save(rewards, file):
    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv(file)


def trainAgent(tries, robot, episodes, teacherAgent=None, feedbackProbability=0, feedbackAccuracy=0, ppl=False):
    if teacherAgent is None:
        filenameFolder = resultsFolder + 'rewardsRL'
    else:
        filenameFolder = resultsFolder + 'rewardsIRL'

    for i in range(tries):
        print('Training agent number: ' + str(i + 1))

        robot_controller.reset()
        rewards = robot_controller.train(episodes)
        # agent.train(episodes, teacherAgent, feedbackProbability, feedbackAccuracy, ppl)
        if teacherAgent is None:
            agentPath = resultsFolder + '/agentRL' + str(i) + '.npy'
            filenameRewards = filenameFolder + str(i) + '.csv'
        else:
            agentPath = resultsFolder + '/agentIRL' + str(i) + '_' + str(feedbackProbability) + '_' + str(
                feedbackAccuracy) + '_' + str(ppl) + '.npy'
            filenameRewards = filenameFolder + str(i) + '_' + str(feedbackProbability) + '_' + str(
                feedbackAccuracy) + '_' + str(ppl) + '.csv'
        robot_controller.save(agentPath)
        save(rewards, filenameRewards)
    return


resultsFolder = 'results/tests/'
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)
    print("Directory ", resultsFolder, " Created ")
timestep = 200
episodes = 1000

INITIAL = (0, 0, 0)
robot_controller = RobotController(timestep, INITIAL)
# robot_controller.load(resultsFolder + 'agentRL0.npy')
trainAgent(1, robot_controller, episodes)

# import matplotlib.pyplot as plt
# import cv2
# import matplotlib.image as mpimg
# cv2.startWindowThread()
# cv2.namedWindow("preview")
# for step in range(timestep):
#     print(step)
#     action = robot_controller.random_action()
#     obs, r, d = robot_controller.step(action)
#
#     distance1, distance2, img = obs

# print(img)
# cv2.imshow("preview", img)
