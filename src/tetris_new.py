#!/usr/bin/python

"""
Python implementation of text-mode version of the Tetris game

Quick play instructions:

 - a (return): move piece left
 - d (return): move piece right
 - w (return): rotate piece counter clockwise
 - s (return): rotate piece clockwise
 - e (return): just move the piece downwards as is

"""
import ast
import os
import random
import sys

from copy import deepcopy

import constants

# DECLARE ALL THE CONSTANTS
BOARD_SIZE = 8
# Extra two are for the walls, playing area will have size as BOARD_SIZE
EFF_BOARD_SIZE = BOARD_SIZE + 2

PIECES = [

    [[1], [1], [1], [1]],

    [[1, 0],
     [1, 0],
     [1, 1]],

    [[0, 1],
     [0, 1],
     [1, 1]],

    [[0, 1],
     [1, 1],
     [1, 0]],

    [[1, 1],
     [1, 1]]

]

# Constants for user input
MOVE_LEFT = 'a'
MOVE_RIGHT = 'd'
ROTATE_ANTICLOCKWISE = 'w'
ROTATE_CLOCKWISE = 's'
NO_MOVE = 'e'
QUIT_GAME = 'q'


class Tetris(object):
    def __init__(self):
        self.piece_pos = None
        self.curr_piece = None
        self.board = None

    def get_board_constraints(self):
        return BOARD_SIZE

    def print_board(self, board, curr_piece, piece_pos, error_message=''):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Details:
        --------
        Prints out the board, piece and playing instructions to STDOUT
        If there are any error messages then prints them to STDOUT as well
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Text mode version of the TETRIS game\n\n")

        board_copy = deepcopy(board)
        curr_piece_size_x = len(curr_piece)
        curr_piece_size_y = len(curr_piece[0])
        for i in range(curr_piece_size_x):
            for j in range(curr_piece_size_y):
                board_copy[piece_pos[0] + i][piece_pos[1] + j] = curr_piece[i][j] | board[piece_pos[0] + i][
                    piece_pos[1] + j]

        # Print the board to STDOUT
        for i in range(EFF_BOARD_SIZE):
            for j in range(EFF_BOARD_SIZE):
                if board_copy[i][j] == 1:
                    print("*", end='')
                else:
                    print(" ", end='')
            print("")

        print("Quick play instructions:\n")
        print(" - a (return): move piece left")
        print(" - d (return): move piece right")
        print(" - w (return): rotate piece counter clockwise")
        print(" - s (return): rotate piece clockwise")

        # In case user doesn't want to alter the position of the piece
        # and he doesn't want to rotate the piece either and just wants to move
        # in the downward direction, he can choose 'f'
        print(" - e (return): just move the piece downwards as is")
        print(" - q (return): to quit the game anytime")

        if error_message:
            print(error_message)
        print("Your move:", )

    def get_board(self):
        board = self.board
        board_copy = deepcopy(board)
        curr_piece_size_x = len(self.curr_piece)
        curr_piece_size_y = len(self.curr_piece[0])
        updated_matrix = [[2 if element == 1 else element for element in row] for row in self.curr_piece]

        for i in range(curr_piece_size_x):
            for j in range(curr_piece_size_y):
                board_copy[self.piece_pos[0] + i][self.piece_pos[1] + j] = updated_matrix[i][j] | \
                                                                           board[self.piece_pos[0] + i][
                                                                               self.piece_pos[1] + j]
        board = [x[1:-1] for x in board_copy[1:-1]]
        return board

    def init_board(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        board - the matrix with the walls of the gameplay
        """
        board = [[0 for x in range(EFF_BOARD_SIZE)] for y in range(EFF_BOARD_SIZE)]
        for i in range(EFF_BOARD_SIZE):
            board[i][0] = 1
        for i in range(EFF_BOARD_SIZE):
            board[EFF_BOARD_SIZE - 1][i] = 1
        for i in range(EFF_BOARD_SIZE):
            board[i][EFF_BOARD_SIZE - 1] = 1

        return board

    def get_random_piece(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        piece - a random piece from the PIECES constant declared above
        """
        idx = random.randrange(len(PIECES))
        return PIECES[idx]

    def get_random_position(self, curr_piece):
        """
        Parameters:
        -----------
        curr_piece - piece which is alive in the game at the moment

        Returns:
        --------
        piece_pos - a randomly (along x-axis) chosen position for this piece
        """
        curr_piece_size = len(curr_piece)

        # This x refers to rows, rows go along y-axis
        x = 0
        # This y refers to columns, columns go along x-axis
        y = random.randrange(1, EFF_BOARD_SIZE - curr_piece_size)
        return [x, y]

    def is_game_over(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board
        Returns:
        --------
        True - if game is over
        False - if game is live and player can still move
        """
        # If the piece cannot move down and the position is still the first row
        # of the board then the game has ended
        if not self.can_move_down(board, curr_piece, piece_pos) and piece_pos[0] == 0:
            return True
        return False

    def get_left_move(self, piece_pos):
        """
        Parameters:
        -----------
        piece_pos - position of piece on the board

        Returns:
        --------
        piece_pos - new position of the piece shifted to the left
        """
        # Shift the piece left by 1 unit
        new_piece_pos = [piece_pos[0], piece_pos[1] - 1]
        return new_piece_pos

    def get_right_move(self, piece_pos):
        """
        Parameters:
        -----------
        piece_pos - position of piece on the board

        Returns:
        --------
        piece_pos - new position of the piece shifted to the right
        """
        # Shift the piece right by 1 unit
        new_piece_pos = [piece_pos[0], piece_pos[1] + 1]
        return new_piece_pos

    def get_down_move(self, piece_pos):
        """
        Parameters:
        -----------
        piece_pos - position of piece on the board

        Returns:
        --------
        piece_pos - new position of the piece shifted downward
        """
        # Shift the piece down by 1 unit
        new_piece_pos = [piece_pos[0] + 1, piece_pos[1]]
        return new_piece_pos

    def rotate_clockwise(self, piece):
        """
        Paramertes:
        -----------
        piece - matrix of the piece to rotate

        Returns:
        --------
        piece - Clockwise rotated piece

        Details:
        --------
        We first reverse all the sub lists and then zip all the sublists
        This will give us a clockwise rotated matrix
        """
        piece_copy = deepcopy(piece)
        reverse_piece = piece_copy[::-1]
        return list(list(elem) for elem in zip(*reverse_piece))

    def rotate_anticlockwise(self, piece):
        """
        Paramertes:
        -----------
        piece - matrix of the piece to rotate

        Returns:
        --------
        Anti-clockwise rotated piece

        Details:
        --------
        If we rotate any piece in clockwise direction for 3 times, we would eventually
        get the piece rotated in anti clockwise direction
        """
        piece_copy = deepcopy(piece)
        # Rotating clockwise thrice will be same as rotating anticlockwise :)
        piece_1 = self.rotate_clockwise(piece_copy)
        piece_2 = self.rotate_clockwise(piece_1)
        return self.rotate_clockwise(piece_2)

    def merge_board_and_piece(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        None

        Details:
        --------
        Fixes the position of the passed piece at piece_pos in the board
        This means that the new piece will now come into the play

        We also remove any filled up rows from the board to continue the gameplay
        as it happends in a tetris game
        """
        reward = 0
        lowest_possible = 0
        for i, row in enumerate(board):
            if any(x == 0 for x in row):
                lowest_possible = max(lowest_possible, i)

        curr_piece_size_x = len(curr_piece)
        curr_piece_size_y = len(curr_piece[0])

        lowest_block = piece_pos[0] + curr_piece_size_x - 1

        ignore_placement = [
            [[0, 1],
             [1, 1],
             [1, 0]],

            [[1, 1, 0],
             [0, 1, 1]],

        ]

        bad_placed = any(x == 0 for x in curr_piece[-1]) and not (
                    curr_piece == ignore_placement[0] or curr_piece == ignore_placement[1])

        amount = self.get_board()[lowest_block - 1].count(2)

        if lowest_possible - lowest_block == 0:
            reward += 1
        else:
            reward -= 2
            # reward += max((3 + lowest_block - lowest_possible), -4)
            # reward += amount
        if bad_placed:
            reward -= 1
        for i in range(curr_piece_size_x):
            for j in range(curr_piece_size_y):
                board[piece_pos[0] + i][piece_pos[1] + j] = curr_piece[i][j] | board[piece_pos[0] + i][piece_pos[1] + j]

        # After merging the board and piece
        # If there are rows which are completely filled then remove those rows

        # Declare empty row to add later
        empty_row = [0] * EFF_BOARD_SIZE
        empty_row[0] = 1
        empty_row[EFF_BOARD_SIZE - 1] = 1

        # Declare a constant row that is completely filled
        filled_row = [1] * EFF_BOARD_SIZE

        # Count the total filled rows in the board
        filled_rows = 0
        for row in board:
            if row == filled_row:
                filled_rows += 1

        # The last row is always a filled row because it is the boundary
        # So decrease the count for that one
        filled_rows -= 1

        for i in range(filled_rows):
            board.remove(filled_row)

        # Add extra empty rows on the top of the board to compensate for deleted rows
        for i in range(filled_rows):
            board.insert(0, empty_row)
        reward += filled_rows * constants.REWARD
        return reward

    def overlap_check(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if piece do not overlap with any other piece or walls
        False - if piece overlaps with any other piece or board walls
        """
        curr_piece_size_x = len(curr_piece)
        curr_piece_size_y = len(curr_piece[0])
        for i in range(curr_piece_size_x):
            for j in range(curr_piece_size_y):
                if board[piece_pos[0] + i][piece_pos[1] + j] == 1 and curr_piece[i][j] == 1:
                    return False
        return True

    def can_move_left(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if we can move the piece left
        False - if we cannot move the piece to the left,
                means it will overlap if we move it to the left
        """
        piece_pos = self.get_left_move(piece_pos)
        return self.overlap_check(board, curr_piece, piece_pos)

    def can_move_right(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if we can move the piece left
        False - if we cannot move the piece to the right,
                means it will overlap if we move it to the right
        """
        piece_pos = self.get_right_move(piece_pos)
        return self.overlap_check(board, curr_piece, piece_pos)

    def can_move_down(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if we can move the piece downwards
        False - if we cannot move the piece to the downward direction
        """
        piece_pos = self.get_down_move(piece_pos)
        return self.overlap_check(board, curr_piece, piece_pos)

    def can_rotate_anticlockwise(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [x,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if we can move the piece anti-clockwise
        False - if we cannot move the piece to anti-clockwise
                might happen in case rotating would overlap with any existing piece
        """
        curr_piece = self.rotate_anticlockwise(curr_piece)
        return self.overlap_check(board, curr_piece, piece_pos)

    def can_rotate_clockwise(self, board, curr_piece, piece_pos):
        """
        Parameters:
        -----------
        board - matrix of the size of the board
        curr_piece - matrix for the piece active in the game
        piece_pos - [Fx,y] co-ordinates of the top-left cell in the piece matrix
                    w.r.t. the board

        Returns:
        --------
        True - if we can move the piece clockwise
        False - if we cannot move the piece to clockwise
                might happen in case rotating would overlap with any existing piece
        """
        curr_piece = self.rotate_clockwise(curr_piece)
        return self.overlap_check(board, curr_piece, piece_pos)

    def reset_game(self):
        self.board = self.init_board()
        self.curr_piece = self.get_random_piece()
        self.piece_pos = self.get_random_position(self.curr_piece)
        # self.print_board(self.board, self.curr_piece, self.piece_pos)

    def player_move(self, input):

        # input = str(input).split(",")
        left = input[0] == 1
        right = input[1] == 1
        rotate = input[2] == 1
        # nothing = all(value == "0" for value in input)
        # print(left, right, rotate)
        reward = 0
        if not self.is_game_over(self.board, self.curr_piece, self.piece_pos):
            do_move_down = True
            if left:
                if self.can_move_left(self.board, self.curr_piece, self.piece_pos):
                    self.piece_pos = self.get_left_move(self.piece_pos)
            if right:
                if self.can_move_right(self.board, self.curr_piece, self.piece_pos):
                    self.piece_pos = self.get_right_move(self.piece_pos)
            if rotate:
                if self.can_rotate_anticlockwise(self.board, self.curr_piece, self.piece_pos):
                    self.curr_piece = self.rotate_anticlockwise(self.curr_piece)
            if do_move_down and self.can_move_down(self.board, self.curr_piece, self.piece_pos):
                self.piece_pos = self.get_down_move(self.piece_pos)

            if not self.can_move_down(self.board, self.curr_piece, self.piece_pos):
                reward = self.merge_board_and_piece(self.board, self.curr_piece, self.piece_pos)
                self.curr_piece = self.get_random_piece()
                self.piece_pos = self.get_random_position(self.curr_piece)
            return reward, False
        else:
            return -1, True
        # self.print_board(self.board, self.curr_piece, self.piece_pos)

    def run(self):
        self.reset_game()
        while True:
            self.player_move([0, 0, 0])

    def play_game(self):

        """
        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Details:
        --------
        - Initializes the game
        - Reads player move from the STDIN
        - Checks for the move validity
        - Continues the gameplay if valid move, else prints out error msg
          without changing the board
        - Fixes the piece position on board if it cannot be moved
        - Pops in new piece on top of the board
        - Quits if no valid moves and possible for a new piece
        - Quits in case user wants to quit

        """

        # Initialize the game board, piece and piece position
        board = self.init_board()
        curr_piece = self.get_random_piece()
        piece_pos = self.get_random_position(curr_piece)
        self.print_board(board, curr_piece, piece_pos)

        # Get player move from STDIN
        player_move = input()
        while (not self.is_game_over(board, curr_piece, piece_pos)):
            ERR_MSG = ""
            do_move_down = False
            if player_move == MOVE_LEFT:
                if self.can_move_left(board, curr_piece, piece_pos):
                    piece_pos = self.get_left_move(piece_pos)
                    do_move_down = True
                else:
                    ERR_MSG = "Cannot move left!"
            elif player_move == MOVE_RIGHT:
                if self.can_move_right(board, curr_piece, piece_pos):
                    piece_pos = self.get_right_move(piece_pos)
                    do_move_down = True
                else:
                    ERR_MSG = "Cannot move right!"
            elif player_move == ROTATE_ANTICLOCKWISE:
                if self.can_rotate_anticlockwise(board, curr_piece, piece_pos):
                    curr_piece = self.rotate_anticlockwise(curr_piece)
                    do_move_down = True
                else:
                    ERR_MSG = "Cannot rotate anti-clockwise !"
            elif player_move == ROTATE_CLOCKWISE:
                if self.can_rotate_clockwise(board, curr_piece, piece_pos):
                    curr_piece = self.rotate_clockwise(curr_piece)
                    do_move_down = True
                else:
                    ERR_MSG = "Cannot rotate clockwise!"
            elif player_move == NO_MOVE:
                do_move_down = True
            elif player_move == QUIT_GAME:
                print("Bye. Thank you for playing!")
                sys.exit(0)
            else:
                ERR_MSG = "That is not a valid move!"

            if do_move_down and self.can_move_down(board, curr_piece, piece_pos):
                piece_pos = self.get_down_move(piece_pos)

            # This means the current piece in the game cannot be moved
            # We have to fix this piece in the board and generate a new piece
            if not self.can_move_down(board, curr_piece, piece_pos):
                self.merge_board_and_piece(board, curr_piece, piece_pos)
                curr_piece = self.get_random_piece()
                piece_pos = self.get_random_position(curr_piece)

            # Redraw board
            self.print_board(board, curr_piece, piece_pos, error_message=ERR_MSG)

            # Get player move from STDIN
            player_move = input()

        print("GAME OVER!")


if __name__ == "__main__":
    tetris = Tetris()
    tetris.play_game()