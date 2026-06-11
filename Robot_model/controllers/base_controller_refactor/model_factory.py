from pathlib import Path

from stable_baselines3 import DQN, PPO

from config import ALGORITHM
from paths import model_path, replay_buffer_path


def make_model(env):
    algorithm = ALGORITHM.upper()
    saved_model = Path(f"{model_path()}.zip")

    if algorithm == "PPO":
        return make_ppo_model(env, saved_model)

    if algorithm == "DQN":
        return make_dqn_model(env, saved_model)

    raise ValueError(f"Unknown algorithm: {ALGORITHM}")


def make_ppo_model(env, saved_model):
    if saved_model.is_file():
        print(f"Loading existing model: {saved_model}", flush=True)
        return PPO.load(model_path(), env=env)

    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.02,
        tensorboard_log="./logs/ppo_pioneer",
    )


def make_dqn_model(env, saved_model):
    if saved_model.is_file():
        print(f"Loading existing model: {saved_model}", flush=True)
        model = DQN.load(model_path(), env=env)
        load_replay_buffer_if_available(model)
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


def load_replay_buffer_if_available(model):
    saved_replay_buffer = Path(replay_buffer_path())
    if saved_replay_buffer.is_file():
        print(f"Loading existing replay buffer: {saved_replay_buffer}", flush=True)
        model.load_replay_buffer(replay_buffer_path())


def save_model(model, label):
    model.save(model_path())
    if isinstance(model, DQN):
        model.save_replay_buffer(replay_buffer_path())
    print(f"Saved {label} at {model.num_timesteps} timesteps", flush=True)
