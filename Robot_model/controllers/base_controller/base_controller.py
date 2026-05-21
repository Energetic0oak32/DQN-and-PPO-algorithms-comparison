from controller import Supervisor

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env

from environment import PioneerAdvancedEnv as PioneerBaseEnv


# Escolha aqui qual algoritmo usar
ALGORITHM = "PPO"
# ALGORITHM = "DQN"
DEBUG_MODE = True


def main():
    robot = Supervisor()

    if(DEBUG_MODE == False):

        env = PioneerBaseEnv(
            robot=robot,
            max_steps=1000,
            sensor_limit=1024.0,
            collision_threshold=0.8,
        )

        # Use isso uma vez para testar se o ambiente está certo.
        # Depois pode comentar, porque ele executa resets e steps aleatórios.
        #check_env(env, warn=True)

        if ALGORITHM == "PPO":
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                gamma=0.99,
                tensorboard_log="./logs/ppo_pioneer",
            )

            model.learn(total_timesteps=50_000)
            model.save("ppo_pioneer_model")

        elif ALGORITHM == "DQN":
            model = DQN(
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

            model.learn(total_timesteps=50_000)
            model.save("dqn_pioneer_model")

        else:
            raise ValueError(f"Algoritmo desconhecido: {ALGORITHM}")

        env.close()
    
    else:
        return 0
