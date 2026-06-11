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

        # mesma ordem dos motores:
        # 0 = front left
        # 1 = front right
        # 2 = back left
        # 3 = back right
        self.speeds = [0.0, 0.0, 0.0, 0.0]

        # one-hot da última ação:
        # [STOP, FORWARD, TURN_LEFT, TURN_RIGHT]
        self.actions = [0.0, 0.0, 0.0, 0.0]
        self.action_handlers = {
            Action.STOP: (self.stop, [1.0, 0.0, 0.0, 0.0]),
            Action.FORWARD: (self.forward, [0.0, 1.0, 0.0, 0.0]),
            Action.TURN_LEFT: (self.turn_left, [0.0, 0.0, 1.0, 0.0]),
            Action.TURN_RIGHT: (self.turn_right, [0.0, 0.0, 0.0, 1.0]),
        }

    def set_speeds(self, front_left, front_right, back_left, back_right):
        self.speeds = [
            front_left,
            front_right,
            back_left,
            back_right,
        ]

        self.motors[0].setVelocity(front_left)
        self.motors[1].setVelocity(front_right)
        self.motors[2].setVelocity(back_left)
        self.motors[3].setVelocity(back_right)

    def stop(self):
        self.set_speeds(0.0, 0.0, 0.0, 0.0)

    def forward(self):
        self.set_speeds(
            FORWARD_SPEED,
            FORWARD_SPEED,
            FORWARD_SPEED,
            FORWARD_SPEED,
        )

    def turn_left(self):
        self.set_speeds(
            -TURN_SPEED,  # front left
            TURN_SPEED,   # front right
            -TURN_SPEED,  # back left
            TURN_SPEED,   # back right
        )

    def turn_right(self):
        self.set_speeds(
            TURN_SPEED,   # front left
            -TURN_SPEED,  # front right
            TURN_SPEED,   # back left
            -TURN_SPEED,  # back right
        )

    def action(self, action):
        action = Action(action)

        handler, one_hot = self.action_handlers[action]
        handler()
        self.actions = one_hot

    def get_normalized_speeds(self):
        max_speed = max(abs(FORWARD_SPEED), abs(TURN_SPEED))

        return [
            speed / max_speed
            for speed in self.speeds
        ]

    def get_last_action(self):
        return self.actions
