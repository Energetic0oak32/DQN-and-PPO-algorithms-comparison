import numpy as np

from actions import Action
from config import FRONT_SENSOR_IDS, LEFT_SENSOR_IDS, NUM_SENSORS, RIGHT_SENSOR_IDS


WHEEL_COUNT = 4


def split_observation(observation):
    """
    observation =
    [sensores] + [velocidades das rodas] + [ação one-hot]
    """

    sensor_obs = observation[:NUM_SENSORS]

    speed_start = NUM_SENSORS
    speed_end = NUM_SENSORS + WHEEL_COUNT
    speed_obs = observation[speed_start:speed_end]

    action_obs = observation[speed_end:speed_end + len(Action)]

    return sensor_obs, speed_obs, action_obs


def get_sensor_metrics(sensor_obs):
    front_obs = sensor_obs[list(FRONT_SENSOR_IDS)]
    left_obs = sensor_obs[list(LEFT_SENSOR_IDS)]
    right_obs = sensor_obs[list(RIGHT_SENSOR_IDS)]

    return {
        "mean": float(np.mean(sensor_obs)),
        "front_mean": float(np.mean(front_obs)),
        "left_mean": float(np.mean(left_obs)),
        "right_mean": float(np.mean(right_obs)),
        "near_ratio": float(np.mean(sensor_obs >= 0.5)),
        "critical_ratio": float(np.mean(sensor_obs >= 0.75)),
    }


def reward(action, observation, dangers, collision, distance_moved):
    action = Action(action)

    if collision:
        return -10.0

    sensor_obs, speed_obs, _ = split_observation(observation)
    sensor_metrics = get_sensor_metrics(sensor_obs)

    # Permanecer sem colisao e positivo, mas progresso real vale mais.
    reward = 0.02

    max_danger = dangers["max"]
    front_danger = dangers["front"]
    left_danger = dangers["left"]
    right_danger = dangers["right"]
    front_mean = sensor_metrics["front_mean"]
    lateral_mean = max(sensor_metrics["left_mean"], sensor_metrics["right_mean"])

    mean_speed = float(np.mean(speed_obs))

    # 1. Premia deslocamento real. Girar no lugar praticamente nao pontua.
    reward += min(distance_moved * 25.0, 0.6)

    # 2. Obstaculos frontais sao perigosos; obstaculos laterais podem apenas
    # indicar que o robo esta seguindo uma parede sem colidir.
    reward -= front_danger * 0.6
    reward -= front_mean * 0.2

    # 3. Proximidade critica em qualquer direcao ainda merece cautela.
    reward -= max(0.0, max_danger - 0.75) * 1.5

    # 4. Recompensa andar para frente quando a frente esta livre.
    if action == Action.FORWARD:
        if front_danger < 0.25:
            reward += 0.45
        elif front_danger < 0.45:
            reward += 0.25
        elif front_danger < 0.65:
            reward += 0.05
        else:
            reward -= 0.7

        # O comando de movimento frontal deve ser vantajoso quando for seguro.
        reward += max(0.0, mean_speed) * 0.2

        # Seguir ao lado de uma parede e continuar livre na frente e util.
        if front_danger < 0.45:
            reward += min(lateral_mean, 0.75) * 0.1

    # 5. Parar e ruim, exceto em situacao de emergencia.
    elif action == Action.STOP:
        reward -= 0.25

        if front_danger > 0.75:
            reward += 0.2

    # 6. Virar e uma manobra de recuperacao quando existe perigo frontal.
    elif action == Action.TURN_LEFT:
        if front_danger > 0.6:
            reward += 0.1
        else:
            reward -= 0.3

        # Só premia escolher o lado mais livre quando há motivo para desviar.
        if front_danger > 0.6:
            if left_danger < right_danger:
                reward += 0.2
            else:
                reward -= 0.1

            reward += max(0.0, sensor_metrics["right_mean"] - sensor_metrics["left_mean"]) * 0.1

    elif action == Action.TURN_RIGHT:
        if front_danger > 0.6:
            reward += 0.1
        else:
            reward -= 0.3

        # Só premia escolher o lado mais livre quando há motivo para desviar.
        if front_danger > 0.6:
            if right_danger < left_danger:
                reward += 0.2
            else:
                reward -= 0.1

            reward += max(0.0, sensor_metrics["left_mean"] - sensor_metrics["right_mean"]) * 0.1

    return float(reward)
