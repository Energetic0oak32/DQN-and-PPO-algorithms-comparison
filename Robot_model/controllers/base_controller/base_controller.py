from pathlib import Path

from controller import Supervisor

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from config import COLLISION_THRESHOLD, MAX_STEPS, SENSOR_LIMIT, SIDE_COLLISION_THRESHOLD
from environment import PioneerEnv


ALGORITHM = "PPO"
TRAINING = True
CHECK_ENV = False
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_INTERVAL = 10_000
MODEL_VERSION = "progress_v1"


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_interval=CHECKPOINT_INTERVAL):
        super().__init__()
        self.save_interval = save_interval
        self.next_checkpoint = save_interval

    def _on_training_start(self):
        self.next_checkpoint = (
            (self.model.num_timesteps // self.save_interval) + 1
        ) * self.save_interval

    def _on_step(self):
        if self.model.num_timesteps >= self.next_checkpoint:
            save_model(self.model, label="checkpoint")
            self.next_checkpoint += self.save_interval
        return True


def make_env(robot):
    return PioneerEnv(
        robot=robot,
        max_steps=MAX_STEPS,
        sensor_limit=SENSOR_LIMIT,
        collision_threshold=COLLISION_THRESHOLD,
        side_collision_threshold=SIDE_COLLISION_THRESHOLD,
    )


def make_model(env):
    algorithm = ALGORITHM.upper()
    saved_model = Path(f"{model_path()}.zip")

    if algorithm == "PPO":
        if saved_model.is_file():
            print(f"Loading existing model: {saved_model}")
            return PPO.load(model_path(), env=env)

        return PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            tensorboard_log="./logs/ppo_pioneer",
        )

    if algorithm == "DQN":
        if saved_model.is_file():
            print(f"Loading existing model: {saved_model}")
            model = DQN.load(model_path(), env=env)
            saved_replay_buffer = Path(replay_buffer_path())
            if saved_replay_buffer.is_file():
                print(f"Loading existing replay buffer: {saved_replay_buffer}")
                model.load_replay_buffer(replay_buffer_path())
            return model

        return DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            tensorboard_log="./logs/dqn_pioneer",
        )

    raise ValueError(f"Unknown algorithm: {ALGORITHM}")


def model_path():
    return f"{ALGORITHM.lower()}_pioneer_{MODEL_VERSION}_model"


def replay_buffer_path():
    return f"{ALGORITHM.lower()}_pioneer_{MODEL_VERSION}_replay_buffer.pkl"


def save_model(model, label):
    model.save(model_path())
    if isinstance(model, DQN):
        model.save_replay_buffer(replay_buffer_path())
    print(f"Saved {label} at {model.num_timesteps} timesteps", flush=True)


def main():
    robot = Supervisor()
    env = make_env(robot)

    if CHECK_ENV:
        check_env(env, warn=True)

    if not TRAINING:
        env.close()
        return

    model = make_model(env)
    print(
        f"Starting training from {model.num_timesteps} timesteps "
        f"for {TOTAL_TIMESTEPS} additional timesteps",
        flush=True,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=False,
            callback=SaveCheckpointCallback(),
        )
        save_model(model, label="final model")
    except KeyboardInterrupt:
        save_model(model, label="interrupted model")
        print("Training interrupted by user", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
