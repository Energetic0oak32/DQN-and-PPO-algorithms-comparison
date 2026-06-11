import numpy as np

from config import (
    COLLISION_THRESHOLD,
    FRONT_SENSOR_IDS,
    LEFT_SENSOR_IDS,
    NUM_SENSORS,
    RIGHT_SENSOR_IDS,
    SIDE_COLLISION_THRESHOLD,
    SPAWN_MAX_START_DANGER,
)


def get_dangers(observation):
    sensor_observation = observation[:NUM_SENSORS]

    return {
        "max": float(np.max(sensor_observation)),
        "front": get_front_danger(sensor_observation),
        "left": get_left_danger(sensor_observation),
        "right": get_right_danger(sensor_observation),
    }


def has_collision(
    dangers,
    collision_threshold=COLLISION_THRESHOLD,
    side_collision_threshold=SIDE_COLLISION_THRESHOLD,
):
    return (
        dangers["front"] >= collision_threshold
        or dangers["max"] >= side_collision_threshold
    )


def is_safe_spawn(dangers):
    return (
        dangers["front"] < SPAWN_MAX_START_DANGER
        and dangers["max"] < COLLISION_THRESHOLD
    )


def get_front_danger(sensor_observation):
    return float(np.max(sensor_observation[list(FRONT_SENSOR_IDS)]))


def get_left_danger(sensor_observation):
    return float(np.max(sensor_observation[list(LEFT_SENSOR_IDS)]))


def get_right_danger(sensor_observation):
    return float(np.max(sensor_observation[list(RIGHT_SENSOR_IDS)]))
