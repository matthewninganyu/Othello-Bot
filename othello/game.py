from __future__ import annotations
from .board import BLACK, WHITE, INIT_BLACK, INIT_WHITE, popcount, apply_move, move_gen, get_moves
import numpy as np
from numba import njit, uint64


class Game:
    def __init__(self):
        self.black_bb = INIT_BLACK
        self.white_bb = INIT_WHITE
        self.current_player = BLACK # Black goes first

        # Stores the board state, the current player, and the move that led to this state (for undoing moves)
        self.board_history = [(self.black_bb, self.white_bb, self.current_player, None)]
    
    @property
    def game_over(self):
        return (move_gen(self.black_bb, self.white_bb) == uint64(0) and move_gen(self.white_bb, self.black_bb) == uint64(0))
    
    @property
    def winner(self):
        if not self.game_over:
            return None
        
        black = popcount(self.black_bb)
        white = popcount(self.white_bb)

        if black > white:
            return BLACK
        elif white > black:
            return WHITE
        else:
            return 0 # Tie
        
    @property
    def legal_moves(self):
        if self.current_player == BLACK:
            return get_moves(self.black_bb, self.white_bb)
        else:
            return get_moves(self.white_bb, self.black_bb)
        
    def make_move(self, move: int):
        # Check game over
        if self.game_over:
            print(f"Game ended! Winner: {self.winner}")
            return
        
        # Validate the move
        if move not in self.legal_moves:
            raise ValueError(f"Illegal move: {move}")
    
        # Get the new board state after applying the move
        if self.current_player == BLACK:
            new_black, new_white = apply_move(self.black_bb, self.white_bb, move)
        else:
            new_white, new_black = apply_move(self.white_bb, self.black_bb, move)

        # Save the current state before making the move
        self.board_history.append((self.black_bb, self.white_bb, self.current_player, move))

        # Update the board and switch player
        self.black_bb = new_black
        self.white_bb = new_white

        if self.current_player == BLACK:
            if move_gen(self.white_bb, self.black_bb) == uint64(0):
                pass
            else:
                self.current_player = WHITE
        else:
            if move_gen(self.black_bb, self.white_bb) == uint64(0):
                pass
            else:
                self.current_player = BLACK

    def print_board(self):
        print("  A B C D E F G H")
        for row in range(8):
            print(row + 1, end=" ")
            for col in range(8):
                bit = row * 8 + col
                if (self.black_bb >> bit) & 1:
                    print("B", end=" ")
                elif (self.white_bb >> bit) & 1:
                    print("W", end=" ")
                else:
                    print(".", end=" ")
            print()

