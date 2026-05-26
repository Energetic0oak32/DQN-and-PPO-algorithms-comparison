import gymnasium as gym
import numpy as np
from gymnasium import spaces

from actions import Action
from config import (
    COLLISION_THRESHOLD,
    FRONT_SENSOR_IDS,
    LEFT_SENSOR_IDS,
    MAX_STEPS,
    NUM_SENSORS,
    RIGHT_SENSOR_IDS,
    SENSOR_LIMIT,
)
from devices import DistanceSensors, Wheels
from rewards import advanced_reward, simple_reward


class PioneerBaseEnv(gym.Env):
    """Gymnasium environment for the Pioneer3-AT Webots controller."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot,
        timestep=None,
        max_steps=MAX_STEPS,
        sensor_limit=SENSOR_LIMIT,
        collision_threshold=COLLISION_THRESHOLD,
    ):
        super().__init__()

        self.robot = robot
        self.timestep = int(timestep or robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.sensor_limit = sensor_limit
        self.collision_threshold = collision_threshold

        self.current_step = 0

        self.distance_sensors = DistanceSensors(robot, self.timestep)
        self.wheels = Wheels(robot)

        self.robot_node = robot.getSelf() if hasattr(robot, "getSelf") else None
        self.initial_translation = None
        self.initial_rotation = None
        if self.robot_node is not None:
            self.initial_translation = self.robot_node.getField("translation").getSFVec3f()
            self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_SENSORS,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.wheels.action(Action.STOP)
        self._restore_robot_pose()
        self.distance_sensors.reset_history()
        self.robot.step(self.timestep)

        observation = self._get_observation()
        dangers = self._get_dangers(observation)

        info = {
            "max_sensor_normalized": dangers["max"],
            "step": self.current_step,
        }
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        action = Action(int(action))
        self.current_step += 1

        self.wheels.action(action)
        simulation_status = self.robot.step(self.timestep)

        observation = self._get_observation()
        dangers = self._get_dangers(observation)
        collision = dangers["max"] >= self.collision_threshold
        reward = self._calculate_reward(action, dangers, collision)

        terminated = bool(collision)
        truncated = self.current_step >= self.max_steps or simulation_status == -1

        info = self._make_info(
            action=action,
            dangers=dangers,
            collision=collision,
            simulation_ok=simulation_status != -1,
        )

        return observation, float(reward), terminated, truncated, info

    def _calculate_reward(self, action, dangers, collision):
        return simple_reward(action, dangers, collision)

    def _make_info(self, action, dangers, collision, simulation_ok):
        return {
            "collision": bool(collision),
            "action": int(action),
            "step": self.current_step,
            "max_sensor": dangers["max"],
            "front_danger": dangers["front"],
            "left_danger": dangers["left"],
            "right_danger": dangers["right"],
            "simulation_ok": simulation_ok,
        }

    def _restore_robot_pose(self):
        if self.robot_node is None:
            return

        self.robot_node.getField("translation").setSFVec3f(self.initial_translation)
        self.robot_node.getField("rotation").setSFRotation(self.initial_rotation)
        self.robot_node.resetPhysics()
        if hasattr(self.robot, "simulationResetPhysics"):
            self.robot.simulationResetPhysics()

    def _get_observation(self):
        raw_values = self.distance_sensors.read_all()
        obs = np.array(raw_values, dtype=np.float32) / self.sensor_limit
        return np.clip(obs, 0.0, 1.0)

    def _get_dangers(self, observation):
        return {
            "max": float(np.max(observation)),
            "front": self._get_front_danger(observation),
            "left": self._get_left_danger(observation),
            "right": self._get_right_danger(observation),
        }

    def _get_front_danger(self, observation):
        return float(np.max(observation[list(FRONT_SENSOR_IDS)]))

    def _get_left_danger(self, observation):
        return float(np.max(observation[list(LEFT_SENSOR_IDS)]))

    def _get_right_danger(self, observation):
        return float(np.max(observation[list(RIGHT_SENSOR_IDS)]))

    def close(self):
        self.wheels.action(Action.STOP)


class PioneerAdvancedEnv(PioneerBaseEnv):
    def _calculate_reward(self, action, dangers, collision):
        return advanced_reward(action, dangers, collision)
