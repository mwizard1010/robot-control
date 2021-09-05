from controller import Supervisor, Receiver
# from controller import Robot
import sys


# from RobotController import RobotController


class SupervisorController:
    def __init__(self, timesteps=32):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("MY_BOT")
        if self.robot_node is None:
            sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
            sys.exit(1)
        self.trans_field = self.robot_node.getField("translation")

        self.timestep = timesteps
        self.receiver = Receiver('receiver')
        self.receiver.enable(self.timestep)

        # self.timestep = timestep
        # self.robot_controller = RobotController(self.timestep)

    def reset(self):
        self.supervisor.simulationReset()
        print(self.trans_field)
        INITIAL = [0, 0, 0]
        self.trans_field.setSFVec3f(INITIAL)

    def start(self):
        # trans_field = self.robot_node.getField("translation")
        # values = trans_field.getSFVec3f()
        # print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
        while self.supervisor.step(self.timestep) != -1:
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                self.receiver.nextPacket()
                if message == 'Reset':
                    print('Receive')
                    self.supervisor.simulationReset()


supervisor = SupervisorController()
supervisor.start()
# supervisor.simulationReset()
# robot_node = supervisor.getFromDef("MY_BOT")
# robot_node.restartController()
# if robot_node is None:
#     sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
#     sys.exit(1)
# trans_field = robot_node.getField("translation")
# INITIAL = [0, 0, 0]
# print(trans_field.setSFVec3f(INITIAL))
