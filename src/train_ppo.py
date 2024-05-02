from stable_baselines3 import PPO
import os
from tetris_env import TetrisEnv
from stable_baselines3.common.vec_env import SubprocVecEnv


def train():
    env = TetrisEnv()
    n_envs = 6
    vec_env = SubprocVecEnv([lambda: env for i in range(n_envs)])

    log_path = os.path.join("training", "logs")

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_path, n_epochs=60, device="cpu")
    for i in range(8):
        model.learn(total_timesteps=60000, progress_bar=True)
        PPO_path = os.path.join("Training", "Saved Models", f"PPO_tetris_7_{i}")
        model.save(PPO_path)


if __name__ == "__main__":
    train()
