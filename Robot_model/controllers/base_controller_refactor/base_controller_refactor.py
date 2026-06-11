from controller import Supervisor
from stable_baselines3.common.env_checker import check_env

from config import (
    CHECK_ENV,
    COLLISION_THRESHOLD,
    MAX_STEPS,
    SENSOR_LIMIT,
    SIDE_COLLISION_THRESHOLD,
    TOTAL_TIMESTEPS,
    TRAINING,
)
from environment import PioneerEnv
from model_factory import make_model, save_model
from training_callbacks import SaveCheckpointCallback


def make_env(robot):
    return PioneerEnv(
        robot=robot,
        max_steps=MAX_STEPS,
        sensor_limit=SENSOR_LIMIT,
        collision_threshold=COLLISION_THRESHOLD,
        side_collision_threshold=SIDE_COLLISION_THRESHOLD,
    )


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
