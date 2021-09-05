from controller import Robot, Emitter, Camera
import random
import numpy as np
import math
from collections import deque
from PIL import Image
import PIL

from PPR import PPR
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
import os
import cv2
import io


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

        self.done = False

        self.emitter = Emitter('emitter')

        self.memory = deque(maxlen=50000)
        self.batch_size = 64
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
        model.add(Dense(24, input_shape=(24, 24, 3), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space()), activation='linear'))
        opt = Nadam(learning_rate=self.alpha)
        model.compile(loss='mse', optimizer=opt)

        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def normal_action(self, state, epsilon=0.1):
        # exploration
        if np.random.random() <= epsilon:
            return self.random_action()
        # exploitation
        else:
            return np.argmax(self.main_network.predict(state))

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
        self.gps.enable(self.time_step)
        self.touchSensor1.enable(self.time_step)
        self.touchSensor2.enable(self.time_step)
        self.touchSensor3.enable(self.time_step)

        self.camera = Camera('camera')
        self.camera.enable(self.time_step)

        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)

    def is_collised(self):
        return (self.touchSensor1.getValue() + self.touchSensor2.getValue() + self.touchSensor3.getValue()) > 0

    def step(self, a):
        if not self.done:
            self.robot.step(timestep)

            leftValue = self.leftDistanceSensor.getValue()
            rightValue = self.rightDistanceSensor.getValue()

            if self.has_obstacle(leftValue, rightValue):
                leftSpeed, rightSpeed = self.speed_at_obstacle(leftValue, rightValue)
            else:
                if a == 0:
                    leftSpeed, rightSpeed = self.goStraight()
                elif a == 1:
                    leftSpeed, rightSpeed = self.moveBack()
                elif a == 2:
                    leftSpeed, rightSpeed = self.turnLeft(leftValue, rightValue)
                elif a == 3:
                    leftSpeed, rightSpeed = self.turnRight(leftValue, rightValue)

            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)

            # set observation .............
            img = self.camera.getImage()
            observations = img
            # observations = leftValue, rightValue, img

            # set reward .............
            reward = 0
            if self.is_collised():
                reward = -10
            else:
                reward = -0.01

            # set done .........
            self.set_done()
            return observations, reward, self.done

        return None, None, self.done

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
            print('Done')
            print(self.current_position)
            print(self.final_position)

    def action_space(self):
        """
        0 : straight
        1: back
        2: left
        3: right
        """
        return [0, 1, 2, 3]

    def random_action(self):
        if not self.is_collised():
            # return random.choice([0, 1, 2, 3])
            if np.random.rand() < 0.8:
                return 0
            else:
                return random.choice([1, 2, 3])
        else:
            print('Collision!')
            return 1

    def goStraight(self):
        return self.max_speed, self.max_speed

    def moveBack(self):
        return -self.max_speed, -self.max_speed / 2

    def turnLeft(self, leftDistance, rightDistance):
        return -(leftDistance / 100), (rightDistance / 100) + 0.5

    def turnRight(self, leftDistance, rightDistance):
        return (leftDistance / 100) + 0.5, -(rightDistance / 100)

    def reset(self):
        self.emitter.send('Reset'.encode())
        # self.current_position = self.init_position
        # print(self.current_position, self.gps.getValues())
        self.done = False

    def train(self, episodes):
        rewards = []
        for ep in range(episodes):
            reward = 0
            self.reset()
            for step in range(self.max_steps):
                if self.done:
                    reward += 10
                    print(step, reward)
                    break
                action = self.random_action()
                obs, r, d = self.step(action)

                if (step + 1) % 5 == 0:
                    if obs is not None:
                        img = obs
                        imgPIL = Image.frombytes('RGBA', (64, 64), img)
                        imgPIL.save(str(ep) + '_' + str(step) + '.png')
                reward += r
            rewards.append(reward)
        return rewards


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
timestep = 150
INITIAL = (0, 0, 0)
robot_controller = RobotController(timestep, INITIAL)
r = robot_controller.train(4)
print(r)

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
