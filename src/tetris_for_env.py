import random
import math
import block
import constants
import numpy as np

class Tetris(object):

    def __init__(self):
        self.bx = constants.BOARD_BLOCK_WIDTH
        self.by = constants.BOARD_BLOCK_HEIGHT
        self.blk_list = []
        self.score = 0

    def apply_action(self, action):
        left = action[0] == 1
        right = action[1] == 1
        spin = action[2] == 1
        if left:
            self.active_block.move(-constants.BWIDTH, 0)