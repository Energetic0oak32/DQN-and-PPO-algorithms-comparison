import gymnasium as gym
import numpy as np
from gymnasium import spaces

from actions import Action
from config import (
    COLLISION_THRESHOLD,
    FRONT_SENSOR_IDS,
    LEFT_SENSOR_IDS,
    MAX_STEPS,
    OBSERVATION_SIZE,
    RIGHT_SENSOR_IDS,
    SENSOR_LIMIT,
    SIDE_COLLISION_THRESHOLD,
    NUM_SENSORS,
    RANDOM_SPAWN,
    SPAWN_X_RANGE,
    SPAWN_Y_RANGE,
    SPAWN_YAW_RANGE,
    SPAWN_MAX_START_DANGER,
    SPAWN_MAX_ATTEMPTS,
    SPAWN_BLOCKED_OBJECT_TYPES,
    SPAWN_MIN_OBJECT_DISTANCE,
)
from devices import DistanceSensors, Wheels
from rewards import reward


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
        self.spawn_blockers = self._get_spawn_blockers()
        if self.robot_node is not None:
            self.initial_translation = self.robot_node.getField("translation").getSFVec3f()
            self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()

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

        max_attempts = SPAWN_MAX_ATTEMPTS if RANDOM_SPAWN else 1
        spawn_attempts = 0

        for spawn_attempts in range(1, max_attempts + 1):
            self.wheels.action(Action.STOP)
            self._restore_robot_pose(random_spawn=RANDOM_SPAWN)
            self.distance_sensors.reset_history()
            self.robot.step(self.timestep)

            observation = self._get_observation()
            dangers = self._get_dangers(observation)

            if not RANDOM_SPAWN or self._is_safe_spawn(dangers):
                break
        else:
            self.wheels.action(Action.STOP)
            self._restore_robot_pose(random_spawn=False)
            self.distance_sensors.reset_history()
            self.robot.step(self.timestep)
            observation = self._get_observation()
            dangers = self._get_dangers(observation)

        self.previous_translation = self._get_robot_translation()
        self.total_distance = 0.0

        info = {
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
        distance_moved = self._get_distance_moved()
        self.total_distance += distance_moved

        observation = self._get_observation()
        dangers = self._get_dangers(observation)
        collision = self._has_collision(dangers)
        reward = self._calculate_reward(
            action,
            observation,
            dangers,
            collision,
            distance_moved,
        )

        terminated = bool(collision)
        truncated = self.current_step >= self.max_steps or simulation_status == -1

        info = self._make_info(
            action=action,
            dangers=dangers,
            collision=collision,
            simulation_ok=simulation_status != -1,
            distance_moved=distance_moved,
        )

        return observation, float(reward), terminated, truncated, info

    def _calculate_reward(self, action, observation, dangers, collision, distance_moved):
        return reward(action, observation, dangers, collision, distance_moved)

    def _make_info(self, action, dangers, collision, simulation_ok, distance_moved):
        return {
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

    def _restore_robot_pose(self, random_spawn=RANDOM_SPAWN):
        if self.robot_node is None:
            return

        translation = list(self.initial_translation)
        rotation = list(self.initial_rotation)

        if random_spawn:
            translation, rotation = self._make_random_spawn_pose()

        self.robot_node.getField("translation").setSFVec3f(translation)
        self.robot_node.getField("rotation").setSFRotation(rotation)

        self.robot_node.resetPhysics()

        if hasattr(self.robot, "simulationResetPhysics"):
            self.robot.simulationResetPhysics()

    def _make_random_spawn_pose(self):
        translation = list(self.initial_translation)
        rotation = list(self.initial_rotation)

        for _ in range(SPAWN_MAX_ATTEMPTS * 10):
            x = float(self.np_random.uniform(SPAWN_X_RANGE[0], SPAWN_X_RANGE[1]))
            y = float(self.np_random.uniform(SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1]))
            yaw = float(self.np_random.uniform(SPAWN_YAW_RANGE[0], SPAWN_YAW_RANGE[1]))

            # O WorldInfo padrao usa ENU: Z eh o eixo vertical
            candidate_translation = [
                x,
                y,
                self.initial_translation[2],
            ]

            if self._is_clear_of_spawn_blockers(candidate_translation):
                translation = candidate_translation
                rotation = [
                    0.0,
                    0.0,
                    1.0,
                    yaw,
                ]
                break

        if not self._is_clear_of_spawn_blockers(translation):
            raise RuntimeError(
                "Could not find a random spawn position clear of blocked objects. "
                "Increase SPAWN_X_RANGE/SPAWN_Y_RANGE or reduce SPAWN_MIN_OBJECT_DISTANCE."
            )

        return translation, rotation

    def _get_spawn_blockers(self):
        if not hasattr(self.robot, "getRoot"):
            return []

        root = self.robot.getRoot()
        if root is None:
            return []

        children = root.getField("children")
        if children is None:
            return []

        return self._find_spawn_blockers(children)

    def _find_spawn_blockers(self, children_field):
        blockers = []

        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)

            if node.getTypeName() in SPAWN_BLOCKED_OBJECT_TYPES:
                translation = self._get_node_translation(node)
                if translation is not None:
                    blockers.append(translation)

            nested_blockers = self._find_nested_spawn_blockers(node)
            blockers.extend(nested_blockers)

        return blockers

    def _find_nested_spawn_blockers(self, node):
        children = node.getField("children")
        if children is None:
            return []

        try:
            return self._find_spawn_blockers(children)
        except Exception:
            return []

    def _get_node_translation(self, node):
        translation_field = node.getField("translation")
        if translation_field is None:
            return None

        try:
            return translation_field.getSFVec3f()
        except Exception:
            return None

    def _is_clear_of_spawn_blockers(self, translation):
        for blocker_translation in self.spawn_blockers:
            distance = float(
                np.linalg.norm(
                    np.array(translation[:2], dtype=np.float32)
                    - np.array(blocker_translation[:2], dtype=np.float32)
                )
            )
            if distance < SPAWN_MIN_OBJECT_DISTANCE:
                return False

        return True

    def _get_robot_translation(self):
        if self.robot_node is None:
            return None
        return self.robot_node.getField("translation").getSFVec3f()

    def _get_distance_moved(self):
        current_translation = self._get_robot_translation()
        if current_translation is None or self.previous_translation is None:
            self.previous_translation = current_translation
            return 0.0

        distance_moved = float(
            np.linalg.norm(
                np.array(current_translation, dtype=np.float32)
                - np.array(self.previous_translation, dtype=np.float32)
            )
        )
        self.previous_translation = current_translation
        return distance_moved

    def _get_observation(self):
        raw_values = self.distance_sensors.read_all()

        sensor_obs = np.array(raw_values, dtype=np.float32) / self.sensor_limit
        sensor_obs = np.clip(sensor_obs, 0.0, 1.0)

        speed_obs = np.array(
            self.wheels.get_normalized_speeds(),
            dtype=np.float32,
        )

        action_obs = np.array(
            self.wheels.get_last_action(),
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

    def _get_dangers(self, observation):
        sensor_observation = observation[:NUM_SENSORS]

        return {
            "max": float(np.max(sensor_observation)),
            "front": self._get_front_danger(sensor_observation),
            "left": self._get_left_danger(sensor_observation),
            "right": self._get_right_danger(sensor_observation),
        }

    def _has_collision(self, dangers):
        return (
            dangers["front"] >= self.collision_threshold
            or dangers["max"] >= self.side_collision_threshold
        )

    def _is_safe_spawn(self, dangers):
        return (
            dangers["front"] < SPAWN_MAX_START_DANGER
            and dangers["max"] < self.collision_threshold
        )

    def _get_front_danger(self, observation):
        return float(np.max(observation[list(FRONT_SENSOR_IDS)]))

    def _get_left_danger(self, observation):
        return float(np.max(observation[list(LEFT_SENSOR_IDS)]))

    def _get_right_danger(self, observation):
        return float(np.max(observation[list(RIGHT_SENSOR_IDS)]))

    def close(self):
        self.wheels.action(Action.STOP)
