HISTORY = 5

NUM_SENSORS = 16
OBSERVATION_SIZE = NUM_SENSORS + 4 + 4 #numero de sensores, + velocidade individual de rodas + ação

FORWARD_SPEED = 3.0
TURN_SPEED = 4.0

SENSOR_LIMIT = 1024.0
COLLISION_THRESHOLD = 0.8
SIDE_COLLISION_THRESHOLD = 0.98
MAX_STEPS = 5000

FRONT_SENSOR_IDS = (2, 3, 4, 5)
LEFT_SENSOR_IDS = (0, 1, 2)
RIGHT_SENSOR_IDS = (5, 6, 7)

RANDOM_SPAWN = True

# Mantem margem para o corpo do robo dentro da arena 5 x 5.
SPAWN_X_RANGE = (-2.0, 2.0)
SPAWN_Y_RANGE = (-2.0, 2.0)

# Valores em radianos -pi a pi;
SPAWN_YAW_RANGE = (-3.14159, 3.14159)
