import numpy as np

from actions import Action
from config import (
    BASE_SURVIVAL_REWARD,
    COLLISION_PENALTY,
    CRITICAL_DANGER_PENALTY,
    CRITICAL_DANGER_THRESHOLD,
    DISTANCE_REWARD_MULTIPLIER,
    BACK_DETECTOR_HIT_PENALTY,
    FORWARD_BLOCKED_PENALTY,
    FORWARD_CAUTION_REWARD,
    FORWARD_CAUTION_THRESHOLD,
    FORWARD_CLEAR_REWARD,
    FORWARD_CLEAR_THRESHOLD,
    FORWARD_DANGER_REWARD,
    FORWARD_DANGER_THRESHOLD,
    FORWARD_SPEED_REWARD_MULTIPLIER,
    FRONT_DANGER_PENALTY,
    FRONT_DETECTOR_HIT_PENALTY,
    FRONT_MEAN_PENALTY,
    FORWARD_INTO_FRONT_HIT_PENALTY,
    FRONT_SENSOR_IDS,
    LEFT_SENSOR_IDS,
    MAX_DISTANCE_REWARD,
    NO_PROGRESS_DISTANCE_THRESHOLD,
    NO_PROGRESS_PENALTY,
    NUM_SENSORS,
    RIGHT_SENSOR_IDS,
    SIDE_DETECTOR_HIT_PENALTY,
    STOP_EMERGENCY_REWARD,
    STOP_PENALTY,
    TURN_FRONT_DANGER_THRESHOLD,
    TURN_SAFE_SIDE_REWARD,
    TURN_SIDE_MEAN_REWARD_MULTIPLIER,
    TURN_NO_PROGRESS_PENALTY,
    TURN_AWAY_FROM_SIDE_HIT_REWARD,
    TURN_INTO_SIDE_HIT_PENALTY,
    TURN_UNSAFE_SIDE_PENALTY,
    TURN_WITH_OBSTACLE_REWARD,
    TURN_WITHOUT_OBSTACLE_PENALTY,
    WALL_FOLLOW_LATERAL_LIMIT,
    WALL_FOLLOW_REWARD_MULTIPLIER,
)


WHEEL_COUNT = 4


def split_observation(observation):
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
        "critical_ratio": float(np.mean(sensor_obs >= CRITICAL_DANGER_THRESHOLD)),
    }


def reward(action, observation, dangers, collision, distance_moved, detector_hits=None):
    action = Action(action)
    detector_hits = detector_hits or {}

    if collision:
        return float(COLLISION_PENALTY)

    sensor_obs, speed_obs, _ = split_observation(observation)
    sensor_metrics = get_sensor_metrics(sensor_obs)

    reward_value = BASE_SURVIVAL_REWARD
    reward_value += movement_reward(distance_moved)
    reward_value += no_progress_penalty(distance_moved)
    reward_value += danger_penalty(dangers, sensor_metrics)
    reward_value += detector_penalty(detector_hits)
    reward_value += action_reward(
        action,
        dangers,
        sensor_metrics,
        speed_obs,
        distance_moved,
        detector_hits,
    )

    return float(reward_value)


def movement_reward(distance_moved):
    return min(distance_moved * DISTANCE_REWARD_MULTIPLIER, MAX_DISTANCE_REWARD)


def no_progress_penalty(distance_moved):
    if distance_moved < NO_PROGRESS_DISTANCE_THRESHOLD:
        return -NO_PROGRESS_PENALTY

    return 0.0


def danger_penalty(dangers, sensor_metrics):
    max_danger = dangers["max"]
    front_danger = dangers["front"]
    front_mean = sensor_metrics["front_mean"]

    penalty = 0.0
    penalty -= front_danger * FRONT_DANGER_PENALTY
    penalty -= front_mean * FRONT_MEAN_PENALTY
    penalty -= max(0.0, max_danger - CRITICAL_DANGER_THRESHOLD) * CRITICAL_DANGER_PENALTY
    return penalty


def detector_penalty(detector_hits):
    penalty = 0.0

    if detector_hits.get("front", False):
        penalty -= FRONT_DETECTOR_HIT_PENALTY
    if detector_hits.get("left", False):
        penalty -= SIDE_DETECTOR_HIT_PENALTY
    if detector_hits.get("right", False):
        penalty -= SIDE_DETECTOR_HIT_PENALTY
    if detector_hits.get("back", False):
        penalty -= BACK_DETECTOR_HIT_PENALTY

    return penalty


def action_reward(action, dangers, sensor_metrics, speed_obs, distance_moved, detector_hits):
    if action == Action.FORWARD:
        return forward_reward(dangers, sensor_metrics, speed_obs, detector_hits)

    if action == Action.STOP:
        return stop_reward(dangers)

    if action == Action.TURN_LEFT:
        return turn_reward(
            front_danger=dangers["front"],
            chosen_side_danger=dangers["left"],
            opposite_side_danger=dangers["right"],
            chosen_side_mean=sensor_metrics["left_mean"],
            opposite_side_mean=sensor_metrics["right_mean"],
            distance_moved=distance_moved,
            chosen_side_hit=detector_hits.get("left", False),
            opposite_side_hit=detector_hits.get("right", False),
        )

    if action == Action.TURN_RIGHT:
        return turn_reward(
            front_danger=dangers["front"],
            chosen_side_danger=dangers["right"],
            opposite_side_danger=dangers["left"],
            chosen_side_mean=sensor_metrics["right_mean"],
            opposite_side_mean=sensor_metrics["left_mean"],
            distance_moved=distance_moved,
            chosen_side_hit=detector_hits.get("right", False),
            opposite_side_hit=detector_hits.get("left", False),
        )

    return 0.0


def forward_reward(dangers, sensor_metrics, speed_obs, detector_hits):
    front_danger = dangers["front"]
    lateral_mean = max(sensor_metrics["left_mean"], sensor_metrics["right_mean"])
    mean_speed = float(np.mean(speed_obs))

    reward_value = 0.0

    if front_danger < FORWARD_CLEAR_THRESHOLD:
        reward_value += FORWARD_CLEAR_REWARD
    elif front_danger < FORWARD_CAUTION_THRESHOLD:
        reward_value += FORWARD_CAUTION_REWARD
    elif front_danger < FORWARD_DANGER_THRESHOLD:
        reward_value += FORWARD_DANGER_REWARD
    else:
        reward_value -= FORWARD_BLOCKED_PENALTY

    reward_value += max(0.0, mean_speed) * FORWARD_SPEED_REWARD_MULTIPLIER

    if front_danger < FORWARD_CAUTION_THRESHOLD:
        reward_value += (
            min(lateral_mean, WALL_FOLLOW_LATERAL_LIMIT)
            * WALL_FOLLOW_REWARD_MULTIPLIER
        )

    if detector_hits.get("front", False):
        reward_value -= FORWARD_INTO_FRONT_HIT_PENALTY

    return reward_value


def stop_reward(dangers):
    reward_value = -STOP_PENALTY

    if dangers["front"] > CRITICAL_DANGER_THRESHOLD:
        reward_value += STOP_EMERGENCY_REWARD

    return reward_value


def turn_reward(
    front_danger,
    chosen_side_danger,
    opposite_side_danger,
    chosen_side_mean,
    opposite_side_mean,
    distance_moved,
    chosen_side_hit,
    opposite_side_hit,
):
    if front_danger <= TURN_FRONT_DANGER_THRESHOLD:
        return -TURN_WITHOUT_OBSTACLE_PENALTY

    reward_value = TURN_WITH_OBSTACLE_REWARD

    if distance_moved < NO_PROGRESS_DISTANCE_THRESHOLD:
        reward_value -= TURN_NO_PROGRESS_PENALTY

    if chosen_side_hit:
        reward_value -= TURN_INTO_SIDE_HIT_PENALTY

    if opposite_side_hit:
        reward_value += TURN_AWAY_FROM_SIDE_HIT_REWARD

    if chosen_side_danger < opposite_side_danger:
        reward_value += TURN_SAFE_SIDE_REWARD
    else:
        reward_value -= TURN_UNSAFE_SIDE_PENALTY

    reward_value += (
        max(0.0, opposite_side_mean - chosen_side_mean)
        * TURN_SIDE_MEAN_REWARD_MULTIPLIER
    )

    return reward_value
