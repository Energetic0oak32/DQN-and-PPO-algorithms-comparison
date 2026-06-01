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


def reward(action, observation, dangers, collision):
    action = Action(action)

    if collision:
        return -10.0

    sensor_obs, speed_obs, _ = split_observation(observation)
    sensor_metrics = get_sensor_metrics(sensor_obs)

    reward = 0.05

    max_danger = dangers["max"]
    front_danger = dangers["front"]
    left_danger = dangers["left"]
    right_danger = dangers["right"]
    front_mean = sensor_metrics["front_mean"]

    mean_speed = float(np.mean(speed_obs))

    # 1. Penalidade por chegar perto demais de qualquer obstáculo.
    reward -= max_danger * 0.6

    # 2. Obstáculo frontal é mais perigoso que lateral.
    reward -= front_danger * 1.0

    # 3. Considera a distribuição dos obstáculos, não apenas os maiores picos.
    reward -= sensor_metrics["mean"] * 0.3
    reward -= front_mean * 0.5
    reward -= sensor_metrics["near_ratio"] * 0.25
    reward -= sensor_metrics["critical_ratio"] * 0.5

    # 4. Recompensa andar para frente quando a frente está livre.
    if action == Action.FORWARD:
        if front_danger < 0.25:
            reward += 1.0
        elif front_danger < 0.45:
            reward += 0.5
        elif front_danger < 0.65:
            reward += 0.1
        else:
            reward -= 1.0

        # Se mandou ir pra frente, mas a velocidade média não está positiva,
        # algo está estranho ou pouco útil.
        reward += max(0.0, mean_speed) * 0.2

    # 5. Parar é ruim, exceto em situação de emergência.
    elif action == Action.STOP:
        reward -= 0.3

        if front_danger > 0.75:
            reward += 0.2

    # 6. Virar é bom quando existe perigo frontal.
    elif action == Action.TURN_LEFT:
        if front_danger > 0.6:
            reward += 0.4
        else:
            reward -= 0.15

        # Só premia escolher o lado mais livre quando há motivo para desviar.
        if front_danger > 0.6:
            if left_danger < right_danger:
                reward += 0.4
            else:
                reward -= 0.2

            reward += max(0.0, sensor_metrics["right_mean"] - sensor_metrics["left_mean"]) * 0.2

    elif action == Action.TURN_RIGHT:
        if front_danger > 0.6:
            reward += 0.4
        else:
            reward -= 0.15

        # Só premia escolher o lado mais livre quando há motivo para desviar.
        if front_danger > 0.6:
            if right_danger < left_danger:
                reward += 0.4
            else:
                reward -= 0.2

            reward += max(0.0, sensor_metrics["left_mean"] - sensor_metrics["right_mean"]) * 0.2

    return float(reward)
