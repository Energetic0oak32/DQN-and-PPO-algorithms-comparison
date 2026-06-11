import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import CHECKPOINT_INTERVAL
from model_factory import save_model


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_interval=CHECKPOINT_INTERVAL):
        super().__init__()
        self.save_interval = save_interval
        self.next_checkpoint = save_interval
        self.window_steps = 0
        self.window_reward = 0.0
        self.window_distance = 0.0
        self.window_collisions = 0
        self.window_actions = [0, 0, 0, 0]

    def _on_training_start(self):
        self.next_checkpoint = (
            (self.model.num_timesteps // self.save_interval) + 1
        ) * self.save_interval

    def _on_step(self):
        self._collect_step_metrics()

        if self.model.num_timesteps >= self.next_checkpoint:
            self._record_window_metrics()
            save_model(self.model, label="checkpoint")
            self.next_checkpoint += self.save_interval
        return True

    def _collect_step_metrics(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        actions = np.asarray(self.locals.get("actions", []), dtype=np.int64).flatten()

        self.window_steps += len(infos)
        self.window_reward += float(np.sum(rewards))

        for action in actions:
            if 0 <= action < len(self.window_actions):
                self.window_actions[int(action)] += 1

        for info in infos:
            self.window_distance += float(info.get("distance_moved", 0.0))
            if info.get("collision", False):
                self.window_collisions += 1

    def _record_window_metrics(self):
        if self.window_steps == 0:
            return

        self.logger.record("custom/mean_step_reward", self.window_reward / self.window_steps)
        self.logger.record("custom/mean_distance_moved", self.window_distance / self.window_steps)
        self.logger.record("custom/collision_rate", self.window_collisions / self.window_steps)

        for action, count in enumerate(self.window_actions):
            self.logger.record(f"custom/action_{action}_rate", count / self.window_steps)

        self.window_steps = 0
        self.window_reward = 0.0
        self.window_distance = 0.0
        self.window_collisions = 0
        self.window_actions = [0, 0, 0, 0]
