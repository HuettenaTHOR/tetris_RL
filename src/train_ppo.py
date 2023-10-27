from stable_baselines3 import PPO
import os
from tetris_env import TetrisEnv
from stable_baselines3.common.vec_env import SubprocVecEnv


def train():
    env = TetrisEnv()
    n_envs = 4
    vec_env = SubprocVecEnv([lambda: env for i in range(n_envs)])


    log_path = os.path.join("training", "logs")
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    """
    episodes = 5
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    """
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_path, n_epochs=2000)
    model.learn(total_timesteps=40000)
    PPO_path = os.path.join("Training", "Saved Models", "PPO_tetris_7")
    model.save(PPO_path)


if __name__ == "__main__":
    train()
