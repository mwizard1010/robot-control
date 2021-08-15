"""my_controller controller."""

from controller import Robot

from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

class RobotController:

    def __init__(self, max_steps=1000, max_speed=2):
        self.robot = Robot()
        # self.timestep = int(robot.getBasicTimeStep())
        self.timestep = max_steps
        self.max_speed = max_speed

        self.setupDevice()

        # self.memory = deque(maxlen=50000)
        # self.batch_size = 64
        # self.alpha = alpha
        # self.gamma = gamma
        # self.epsilon = epsilon
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_log_decay

        # Interactive
        # self.feedbackAmount = 0
        #
        # # model
        # self.main_network = self.build_network()
        # self.target_network = self.build_network()
        # self.update_target()

    def setupDevice(self):
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        self.rightDistanceSensor = self.robot.getDevice('ds1')
        self.leftDistanceSensor = self.robot.getDevice('ds0')
        self.gps = self.robot.getDevice('gps')
        self.touchSensor = self.robot.getDevice('touch_sensor')

        self.rightDistanceSensor.enable(self.timestep)
        self.leftDistanceSensor.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.touchSensor.enable(self.timestep)

        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)

    def run(self):
        while self.robot.step(self.timestep) != -1:
            leftValue = self.leftDistanceSensor.getValue()
            rightValue = self.rightDistanceSensor.getValue()

            leftSpeed = 0
            rightSpeed = 0
            if leftValue > 500:
                if rightValue > 500:
                    leftSpeed, rightSpeed = self.moveBack()
                else:
                    leftSpeed, rightSpeed = self.turnRight(leftValue, rightValue)
            elif rightValue > 500:
                leftSpeed, rightSpeed = self.turnLeft(leftValue, rightValue)
            else:
                leftSpeed, rightSpeed = self.goStraight()

            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)
            print('Distance', leftValue, rightValue)
            print('Speed', leftSpeed, rightSpeed)
            if self.touchSensor.getValue() > 0:
                print('Collision!')
            # print('Distance:', leftValue, rightValue)
            # print('GPS:', self.gps.getValues())
            pass

    def moveBack(self):
        return -self.max_speed, -self.max_speed / 2

    def turnLeft(self, leftDistance, rightDistance):
        return -(leftDistance / 100), (rightDistance / 100) + 0.5

    def turnRight(self, leftDistance, rightDistance):
        return (leftDistance / 100) + 0.5, -(rightDistance / 100)

    def goStraight(self):
        return self.max_speed, self.max_speed


robot_controller = RobotController()
robot_controller.run()
