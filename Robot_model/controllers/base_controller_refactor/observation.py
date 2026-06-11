import numpy as np


def build_observation(distance_sensors, wheels, sensor_limit):
    raw_values = distance_sensors.read_all()

    sensor_obs = np.array(raw_values, dtype=np.float32) / sensor_limit
    sensor_obs = np.clip(sensor_obs, 0.0, 1.0)

    speed_obs = np.array(
        wheels.get_normalized_speeds(),
        dtype=np.float32,
    )

    action_obs = np.array(
        wheels.get_last_action(),
        dtype=np.float32,
    )

    obs = np.concatenate(
        [
            sensor_obs,
            speed_obs,
            action_obs,
        ]
    )

    return obs.astype(np.float32)


def get_robot_translation(robot_node):
    if robot_node is None:
        return None
    return robot_node.getField("translation").getSFVec3f()


def get_distance_moved(robot_node, previous_translation):
    current_translation = get_robot_translation(robot_node)
    if current_translation is None or previous_translation is None:
        return 0.0, current_translation

    distance_moved = float(
        np.linalg.norm(
            np.array(current_translation, dtype=np.float32)
            - np.array(previous_translation, dtype=np.float32)
        )
    )
    return distance_moved, current_translation
