from controller import Supervisor

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from config import COLLISION_THRESHOLD, MAX_STEPS, SENSOR_LIMIT
from environment import PioneerAdvancedEnv, PioneerBaseEnv


ENV_TYPE = "advanced"
ALGORITHM = "PPO"
TRAINING = True
CHECK_ENV = False
TOTAL_TIMESTEPS = 50_000


def make_env(robot):
    env_classes = {
        "base": PioneerBaseEnv,
        "advanced": PioneerAdvancedEnv,
    }

    try:
        env_class = env_classes[ENV_TYPE.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown environment type: {ENV_TYPE}") from exc

    return env_class(
        robot=robot,
        max_steps=MAX_STEPS,
        sensor_limit=SENSOR_LIMIT,
        collision_threshold=COLLISION_THRESHOLD,
    )


def make_model(env):
    algorithm = ALGORITHM.upper()

    if algorithm == "PPO":
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
    return f"{ALGORITHM.lower()}_pioneer_model"


def main():
    robot = Supervisor()
    env = make_env(robot)

    if CHECK_ENV:
        check_env(env, warn=True)

    if not TRAINING:
        env.close()
        return

    model = make_model(env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path())

    env.close()


if __name__ == "__main__":
    main()
