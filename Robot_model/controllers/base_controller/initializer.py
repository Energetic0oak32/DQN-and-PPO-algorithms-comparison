#settings
HISTORY = 5
FORWARD_SPEED = 3.0
TURN_SPEED = 4.0

def consoleClear():
    print("\033[H\033[2J", end="")

class DistanceSensors:
    def __init__(self, robot, timestep):

        self.sensors = []
        self.names = []
        self.raw_values = []
        self.avg_values = []

        for i in range(16):

            self.raw_values.append([]) #cria a lista [[], [], []...] com acesso de self.list[0] -> [[0], []]
            self.names.append("so" + str(i))
            self.sensors.append(robot.getDevice(self.names[i]))

        for sensor in self.sensors:

            sensor.enable(timestep)

    def reset_history(self):
        self.raw_values = [[] for _ in self.sensors]
        self.avg_values = []

    def read_all(self): 
        for i in range(len(self.sensors)):

            tmp = self.sensors[i].getValue()

            self.raw_values[i].append(tmp)

            if len(self.raw_values[i]) > HISTORY:
                self.raw_values[i].pop(0)

        self.avg_values.clear()

        for i in range(len(self.raw_values)):
            self.avg_values.append(sum(self.raw_values[i])/len(self.raw_values[i]))
        
        return self.avg_values
    
class Wheels:
    def __init__(self, robot):
        self.names = [
            "front left wheel",
            "front right wheel",
            "back left wheel",
            "back right wheel"
        ]

        self.motors = []
        
        for name in self.names:
            motor = robot.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(0)
            self.motors.append(motor)

    def action(self, opc):
        if opc == 0:
            for motor in self.motors:
                motor.setVelocity(0)

        elif opc == 1:
            for motor in self.motors:
                motor.setVelocity(FORWARD_SPEED)
        
        elif opc == 2:
            self.motors[0].setVelocity(-TURN_SPEED)
            self.motors[2].setVelocity(-TURN_SPEED)
            self.motors[1].setVelocity(TURN_SPEED)
            self.motors[3].setVelocity(TURN_SPEED)

        elif opc == 3:
            self.motors[0].setVelocity(TURN_SPEED)
            self.motors[2].setVelocity(TURN_SPEED)
            self.motors[1].setVelocity(-TURN_SPEED)
            self.motors[3].setVelocity(-TURN_SPEED)

            
                


