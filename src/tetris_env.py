import gym
import numpy as np
from gym import Env
import constants
from gym.spaces import Discrete, Box, Dict, MultiBinary

from tetris_new import Tetris
from colorama import Fore


class TetrisEnv(Env):
    def __init__(self):
        self.tetris = Tetris()
        self.action_space = Box(low=0, high=1, shape=(3,), dtype=np.int32)
        # self.action_space = MultiBinary(3)
        BOARD_SIZE = self.tetris.get_board_constraints()
        self.observation_space = Box(low=0, high=2, shape=(BOARD_SIZE, BOARD_SIZE),
                                     dtype=np.int32)
        self.game_length = constants.GAME_LENGTH_FOR_ENV
        self.tetris.reset_game()
        self.state = self.tetris.get_board()

    def step(self, action):
        action = [0 if x == 0 else 1 for x in action]
        reward, end = self.tetris.player_move(action)
        self.game_length -= 1
        self.state = self.tetris.get_board()
        # reward, end = self.tetris.get_reward()

        if self.game_length <= 0 or end:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        for row in self.state:
            print(Fore.GREEN + "#, ", end="")
            for item in row:
                if item == 0:
                    print(Fore.WHITE + "0, ", end="")
                if item == 1:
                    print(Fore.BLUE + "1, ", end="")
                if item == 2:
                    print(Fore.RED + "2, ", end="")
            print(Fore.GREEN + "#")

    def reset(self):
        self.tetris = Tetris()
        self.tetris.reset_game()
        self.state = self.tetris.get_board()
        self.game_length = constants.GAME_LENGTH_FOR_ENV
        return self.state


def run():
    env = TetrisEnv()

    episodes = 5
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()

            # action = env.action_space.sample()
            action = [0, 0, 1]
            n_state, reward, done, info = env.step(action)
            # print(reward)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))

if __name__ == "__main__":
    run()