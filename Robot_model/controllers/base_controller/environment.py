import gymnasium as gym
import numpy as np
from gymnasium import spaces

from initializer import DistanceSensors, Wheels


class PioneerBaseEnv(gym.Env):
    """Gymnasium environment for the Pioneer3-AT Webots controller."""
    
    metadata = {"render_modes": []}
    
    FRONT_SENSOR_IDS = (2, 3, 4, 5)
    LEFT_SENSOR_IDS  = (0, 1, 2)
    RIGHT_SENSOR_IDS = (5, 6, 7)

    def __init__(self, robot, timestep=None, max_steps=1000,
                 sensor_limit=1024.0, collision_threshold=0.8):
        super().__init__()

        self.robot = robot
        self.timestep = int(timestep or robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.sensor_limit = sensor_limit
        self.collision_threshold = collision_threshold

        self.current_step = 0

        self.distance_sensors = DistanceSensors(robot, self.timestep)
        self.wheels = Wheels(robot)

        # Supervisor (para reset de pose)
        self.robot_node = robot.getSelf() if hasattr(robot, "getSelf") else None
        self.initial_translation = None
        self.initial_rotation = None
        if self.robot_node is not None:
            self.initial_translation = self.robot_node.getField("translation").getSFVec3f()
            self.initial_rotation   = self.robot_node.getField("rotation").getSFRotation()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. Para o robô
        self.wheels.action(0)

        # 2. Restaura posição inicial (se for Supervisor)
        self._restore_robot_pose()

        # 3. Limpa histórico da média móvel dos sensores
        self.distance_sensors.reset_history()

        # 4. Dá um passo para atualizar sensores
        self.robot.step(self.timestep)

        # 5. Obtém observação inicial
        observation = self._get_observation()

        info = {
            "max_sensor_normalized": float(np.max(observation)),
            "step": self.current_step,
        }
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Ação {action} inválida")

        self.current_step += 1
        action = int(action)

        self.wheels.action(action)
        simulation_status = self.robot.step(self.timestep)

        observation = self._get_observation()
        collision = bool(np.max(observation) >= self.collision_threshold)

        # ========== REWARD SIMPLES (PioneerBaseEnv) ==========
        if collision:
            reward = -1.0
        else:
            reward = 0.1
            front_danger = self._get_front_danger(observation)

            if action == 1:  # Forward
                if front_danger < 0.5:
                    reward += 0.2
                elif front_danger < 0.7:
                    reward += 0.05
                else:
                    reward -= 0.3
            elif action == 0:  # Stop
                reward -= 0.05
            elif action in (2, 3):  # Turn
                if front_danger > 0.6:
                    reward += 0.15
                else:
                    reward -= 0.05

        terminated = collision
        truncated = self.current_step >= self.max_steps
        if simulation_status == -1:
            truncated = True

        info = {
            "collision": collision,
            "action": action,
            "step": self.current_step,
            "max_sensor": float(np.max(observation)),
            "front_danger": self._get_front_danger(observation),
            "simulation_ok": simulation_status != -1,
        }

        return observation, float(reward), terminated, truncated, info

    def _restore_robot_pose(self):
        if self.robot_node is None:
            return
        self.robot_node.getField("translation").setSFVec3f(self.initial_translation)
        self.robot_node.getField("rotation").setSFRotation(self.initial_rotation)
        self.robot_node.resetPhysics()
        if hasattr(self.robot, "simulationResetPhysics"):
            self.robot.simulationResetPhysics()

    def _get_observation(self):
        """Versão vetorizada e eficiente."""
        raw_values = self.distance_sensors.read_all()
        obs = np.array(raw_values, dtype=np.float32) / self.sensor_limit
        return np.clip(obs, 0.0, 1.0)

    def _get_front_danger(self, observation):
        return float(np.max(observation[list(self.FRONT_SENSOR_IDS)]))

    def _get_left_danger(self, observation):
        return float(np.max(observation[list(self.LEFT_SENSOR_IDS)]))

    def _get_right_danger(self, observation):
        return float(np.max(observation[list(self.RIGHT_SENSOR_IDS)]))

    def close(self):
        self.wheels.action(0)


# ==================== VERSÃO AVANÇADA (REWARD MAIS RICA) ====================
class PioneerAdvancedEnv(PioneerBaseEnv):
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Ação inválida: {action}")

        self.current_step += 1
        action = int(action)

        self.wheels.action(action)
        simulation_status = self.robot.step(self.timestep)

        observation = self._get_observation()
        max_sensor = np.max(observation)
        collision = bool(max_sensor >= self.collision_threshold)
        front_danger = self._get_front_danger(observation)
        left_danger = self._get_left_danger(observation)
        right_danger = self._get_right_danger(observation)

        if collision:
            reward = -10.0
        else:
            reward = 0.05
            danger_ratio = max_sensor
            reward -= danger_ratio * 0.5
            reward -= front_danger * 0.8

            if action == 1:  # Forward
                if front_danger < 0.3:
                    reward += 0.6
                elif front_danger < 0.5:
                    reward += 0.3
                elif front_danger < 0.7:
                    reward += 0.1
                else:
                    reward -= 0.5
            elif action == 0:  # Stop
                reward -= 0.1
            elif action in (2, 3):  # Turn
                if front_danger > 0.6:
                    reward += 0.4
                    if (action == 2 and right_danger > left_danger) or \
                       (action == 3 and left_danger > right_danger):
                        reward += 0.3
                else:
                    reward -= 0.1

        terminated = collision
        truncated = self.current_step >= self.max_steps
        if simulation_status == -1:
            truncated = True

        info = {
            "collision": collision,
            "action": action,
            "step": self.current_step,
            "max_sensor": float(max_sensor),
            "front_danger": front_danger,
            "left_danger": left_danger,
            "right_danger": right_danger,
            "simulation_ok": simulation_status != -1,
        }

        return observation, float(reward), terminated, truncated, info
