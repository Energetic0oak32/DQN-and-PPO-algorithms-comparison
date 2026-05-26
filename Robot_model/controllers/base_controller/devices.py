from actions import Action
from config import FORWARD_SPEED, HISTORY, NUM_SENSORS, TURN_SPEED


def consoleClear():
    print("\033[H\033[2J", end="")


class DistanceSensors:
    def __init__(self, robot, timestep):
        self.sensors = []
        self.names = []
        self.raw_values = []
        self.avg_values = []

        for i in range(NUM_SENSORS):
            self.raw_values.append([])
            self.names.append(f"so{i}")
            self.sensors.append(robot.getDevice(self.names[i]))

        for sensor in self.sensors:
            sensor.enable(timestep)

    def reset_history(self):
        self.raw_values = [[] for _ in self.sensors]
        self.avg_values = []

    def read_all(self):
        for i, sensor in enumerate(self.sensors):
            value = sensor.getValue()
            self.raw_values[i].append(value)

            if len(self.raw_values[i]) > HISTORY:
                self.raw_values[i].pop(0)

        self.avg_values.clear()

        for values in self.raw_values:
            self.avg_values.append(sum(values) / len(values))

        return self.avg_values


class Wheels:
    def __init__(self, robot):
        self.names = [
            "front left wheel",
            "front right wheel",
            "back left wheel",
            "back right wheel",
        ]

        self.motors = []

        for name in self.names:
            motor = robot.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(0)
            self.motors.append(motor)

    def stop(self):
        for motor in self.motors:
            motor.setVelocity(0)

    def forward(self):
        for motor in self.motors:
            motor.setVelocity(FORWARD_SPEED)

    def turn_left(self):
        self.motors[0].setVelocity(-TURN_SPEED)
        self.motors[2].setVelocity(-TURN_SPEED)
        self.motors[1].setVelocity(TURN_SPEED)
        self.motors[3].setVelocity(TURN_SPEED)

    def turn_right(self):
        self.motors[0].setVelocity(TURN_SPEED)
        self.motors[2].setVelocity(TURN_SPEED)
        self.motors[1].setVelocity(-TURN_SPEED)
        self.motors[3].setVelocity(-TURN_SPEED)

    def action(self, action):
        action = Action(action)

        if action == Action.STOP:
            self.stop()
        elif action == Action.FORWARD:
            self.forward()
        elif action == Action.TURN_LEFT:
            self.turn_left()
        elif action == Action.TURN_RIGHT:
            self.turn_right()
