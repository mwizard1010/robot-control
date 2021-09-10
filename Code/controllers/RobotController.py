from controller import Robot, Emitter, Receiver, Camera
import random
import numpy as np
import math
import struct
import pickle
from collections import deque
from PIL import Image

# from PPR import PPR
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os


class RobotController:
    def __init__(self, max_steps=2, init_position=(0, 0, 0), final_position=(-0.3, 0, 0.3), max_speed=3,
                 ):
        self.robot = Robot()

        self.timestep = int(self.robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.max_speed = max_speed

        self.setupDevice()

        self.init_position = init_position
        self.current_position = init_position
        self.final_position = final_position

        self.done = False

        # Interactive
        self.feedbackAmount = 0

        # self.policy_reuse = PPR()

    def setupDevice(self):
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        self.rightDistanceSensor = self.robot.getDevice('ds1')
        self.leftDistanceSensor = self.robot.getDevice('ds0')
        self.rightDistanceSensor.enable(self.timestep)
        self.leftDistanceSensor.enable(self.timestep)

        self.gps = self.robot.getDevice('gps')
        self.touchSensor1 = self.robot.getDevice('touch_sensor1')
        self.touchSensor2 = self.robot.getDevice('touch_sensor2')
        self.touchSensor3 = self.robot.getDevice('touch_sensor3')
        self.touchSensor4 = self.robot.getDevice('touch_sensor4')
        self.touchSensor5 = self.robot.getDevice('touch_sensor5')
        self.gps.enable(self.timestep)
        self.touchSensor1.enable(self.timestep)
        self.touchSensor2.enable(self.timestep)
        self.touchSensor3.enable(self.timestep)
        self.touchSensor4.enable(self.timestep)
        self.touchSensor5.enable(self.timestep)

        self.camera = Camera('camera')
        self.camera.enable(self.timestep)

        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)

        self.init_leftValue = self.leftDistanceSensor.getValue()
        self.init_rightValue = self.rightDistanceSensor.getValue()

        self.receiver = Receiver('receiver')
        self.emitter = Emitter('emitter')
        self.receiver.enable(self.timestep)

    def is_collised(self):
        if (self.touchSensor1.getValue() + self.touchSensor2.getValue() +
                self.touchSensor3.getValue() + self.touchSensor4.getValue() +
                self.touchSensor5.getValue() > 0):
            print(1, self.touchSensor1.getValue())
            print(2, self.touchSensor2.getValue())
            print(3, self.touchSensor3.getValue())
            print(4, self.touchSensor4.getValue())
            print(5, self.touchSensor5.getValue())
            return True
        gpsValue = self.gps.getValues()
        self.current_position = gpsValue
        if (self.current_position[0] < - 0.5 or self.current_position[0] > 0.5 or
                self.current_position[2] < - 0.5 or self.current_position[2] > 0.5):
            return True

        return False

    def step(self, a):
        if not self.done:
            self.robot.step(self.timestep)
            if not self.is_collised():
                leftValue = self.leftDistanceSensor.getValue()
                rightValue = self.rightDistanceSensor.getValue()
                reward = -0.1
                leftSpeed, rightSpeed = 0, 0
                if a == 0:
                    leftSpeed, rightSpeed = self.turnLeft(leftValue, rightValue)
                elif a == 1:
                    leftSpeed, rightSpeed = self.turnRight(leftValue, rightValue)
                elif a == 2:
                    leftSpeed, rightSpeed = self.goStraight()
                    reward = 0
                # elif a == 3:
                #     leftSpeed, rightSpeed = self.goSlow()

                self.leftMotor.setVelocity(leftSpeed)
                self.rightMotor.setVelocity(rightSpeed)

                # set observation .............

                observations = leftValue, rightValue

                # set reward .............

                # set done .........
                r = self.set_done()
                return observations, reward + r, self.done, False

            else:
                observations = self.reset()
                reward = -100
                return observations, reward, False, True

        return None, None, self.done, False

    def set_done(self):
        gpsValue = self.gps.getValues()
        self.current_position = gpsValue
        if abs(self.current_position[0] - self.final_position[0]) <= 0.08 and \
                abs(self.current_position[2] - self.final_position[2]) <= 0.08:
            self.done = True
            return 1000
        return 0

    def random_action(self):
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
        self.done = False
        return self.init_leftValue, self.init_rightValue

    def send_to_super(self, message, data):
        data = message, data
        dataSerialized = pickle.dumps(data)
        self.emitter.send(dataSerialized)

    def receive_handle(self):
        if self.receiver.getQueueLength() > 0:
            data = self.receiver.getData()
            message, action, step = pickle.loads(data)
            self.receiver.nextPacket()
            if message == 'step':
                obs, r, d, i = self.step(action)
                data = obs, r, d, i, step
                self.send_to_super('step_done', data)
            if message == 'reset':
                obs = self.reset()
                self.send_to_super('reset_done', obs)

        return -1

    def start(self):
        while self.robot.step(self.timestep) != -1:
            self.receive_handle()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

robot_controller = RobotController()
robot_controller.start()
