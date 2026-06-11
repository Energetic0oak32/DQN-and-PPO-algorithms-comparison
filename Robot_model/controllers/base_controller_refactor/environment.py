import gymnasium as gym
import numpy as np
from gymnasium import spaces

from actions import Action
from config import (
    COLLISION_THRESHOLD,
    MAX_STEPS,
    OBSERVATION_SIZE,
    RANDOM_SPAWN,
    SENSOR_LIMIT,
    SIDE_COLLISION_THRESHOLD,
    SPAWN_MAX_ATTEMPTS,
)
from danger import get_dangers, is_safe_spawn
from detectors import DetectorManager
from devices import DistanceSensors, Wheels
from observation import build_observation, get_distance_moved, get_robot_translation
from rewards import reward
from spawn import SpawnManager


class PioneerEnv(gym.Env):
    """Gymnasium environment for the Pioneer3-AT Webots controller."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot,
        timestep=None,
        max_steps=MAX_STEPS,
        sensor_limit=SENSOR_LIMIT,
        collision_threshold=COLLISION_THRESHOLD,
        side_collision_threshold=SIDE_COLLISION_THRESHOLD,
    ):
        super().__init__()

        self.robot = robot
        self.timestep = int(timestep or robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.sensor_limit = sensor_limit
        self.collision_threshold = collision_threshold
        self.side_collision_threshold = side_collision_threshold

        self.current_step = 0

        self.distance_sensors = DistanceSensors(robot, self.timestep)
        self.wheels = Wheels(robot)

        self.robot_node = robot.getSelf() if hasattr(robot, "getSelf") else None
        self.initial_translation = None
        self.initial_rotation = None
        self.previous_translation = None
        self.total_distance = 0.0
        self.spawn_manager = None
        self.detectors = DetectorManager(robot)

        if self.robot_node is not None:
            self.initial_translation = self.robot_node.getField("translation").getSFVec3f()
            self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()
            self.spawn_manager = SpawnManager(
                robot=robot,
                robot_node=self.robot_node,
                initial_translation=self.initial_translation,
                initial_rotation=self.initial_rotation,
            )

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        observation, dangers, detector_hits, spawn_attempts = self._reset_until_safe_spawn()
        self.previous_translation = self._get_robot_translation()
        self.total_distance = 0.0

        info = {
            **self._make_detector_info(detector_hits),
            "max_sensor_normalized": dangers["max"],
            "step": self.current_step,
            "spawn_attempts": spawn_attempts,
        }
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        action = Action(int(action))
        self.current_step += 1

        self.wheels.action(action)
        simulation_status = self.robot.step(self.timestep)
        distance_moved = self._update_distance_moved()
        self.total_distance += distance_moved

        observation = self._get_observation()
        dangers = self._get_dangers(observation)
        detector_hits = self._get_detector_hits()
        collision = self._has_collision(detector_hits)
        reward_value = self._calculate_reward(
            action,
            observation,
            dangers,
            collision,
            distance_moved,
            detector_hits,
        )

        terminated = bool(collision)
        truncated = self.current_step >= self.max_steps or simulation_status == -1

        info = self._make_info(
            action=action,
            dangers=dangers,
            collision=collision,
            simulation_ok=simulation_status != -1,
            distance_moved=distance_moved,
            detector_hits=detector_hits,
        )

        return observation, float(reward_value), terminated, truncated, info

    def _reset_until_safe_spawn(self):
        max_attempts = SPAWN_MAX_ATTEMPTS if RANDOM_SPAWN else 1
        spawn_attempts = 0

        for spawn_attempts in range(1, max_attempts + 1):
            observation, dangers, detector_hits = self._reset_once(random_spawn=RANDOM_SPAWN)

            if not RANDOM_SPAWN or self._is_safe_reset(dangers, detector_hits):
                return observation, dangers, detector_hits, spawn_attempts

        observation, dangers, detector_hits = self._reset_once(random_spawn=False)
        return observation, dangers, detector_hits, spawn_attempts

    def _reset_once(self, random_spawn):
        self.wheels.action(Action.STOP)
        self._restore_robot_pose(random_spawn=random_spawn)
        self.distance_sensors.reset_history()
        self.robot.step(self.timestep)

        observation = self._get_observation()
        dangers = self._get_dangers(observation)
        detector_hits = self._get_detector_hits()
        return observation, dangers, detector_hits

    def _calculate_reward(
        self,
        action,
        observation,
        dangers,
        collision,
        distance_moved,
        detector_hits,
    ):
        return reward(
            action,
            observation,
            dangers,
            collision,
            distance_moved,
            detector_hits,
        )

    def _make_info(
        self,
        action,
        dangers,
        collision,
        simulation_ok,
        distance_moved,
        detector_hits,
    ):
        return {
            **self._make_detector_info(detector_hits),
            "collision": bool(collision),
            "action": int(action),
            "step": self.current_step,
            "max_sensor": dangers["max"],
            "front_danger": dangers["front"],
            "left_danger": dangers["left"],
            "right_danger": dangers["right"],
            "simulation_ok": simulation_ok,
            "distance_moved": distance_moved,
            "total_distance": self.total_distance,
        }

    def _make_detector_info(self, detector_hits):
        return {
            f"{name}_detector_hit": bool(hit)
            for name, hit in detector_hits.items()
        }

    def _restore_robot_pose(self, random_spawn=RANDOM_SPAWN):
        if self.spawn_manager is None:
            return

        self.spawn_manager.restore_robot_pose(
            np_random=self.np_random,
            random_spawn=random_spawn,
        )

    def _get_robot_translation(self):
        return get_robot_translation(self.robot_node)

    def _update_distance_moved(self):
        distance_moved, current_translation = get_distance_moved(
            self.robot_node,
            self.previous_translation,
        )
        self.previous_translation = current_translation
        return distance_moved

    def _get_observation(self):
        return build_observation(
            distance_sensors=self.distance_sensors,
            wheels=self.wheels,
            sensor_limit=self.sensor_limit,
        )

    def _get_dangers(self, observation):
        return get_dangers(observation)

    def _get_detector_hits(self):
        return self.detectors.get_hits()

    def _has_collision(self, detector_hits):
        return bool(detector_hits.get("center", False))

    def _is_safe_reset(self, dangers, detector_hits):
        return is_safe_spawn(dangers) and not detector_hits.get("center", False)

    def close(self):
        self.wheels.action(Action.STOP)
