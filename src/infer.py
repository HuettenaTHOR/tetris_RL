import tetris_env
from stable_baselines3 import PPO
import os
import time
from termcolor import colored

def infer():
    env = tetris_env.TetrisEnv()
    model_path = os.path.join("training", "Saved Models", "PPO_tetris_7")
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render()

        action, _ = model.predict(obs)
        new_action = [0 if x == 0 else 1 for x in action]
        print(new_action)
        obs, reward, done, info = env.step(new_action)

        time.sleep(0.2)
        score += reward
    print(f"score: {score}")


if __name__ == "__main__":
    infer()