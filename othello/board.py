"""
game.py - Othello engine
----------------------------------------------
Board Rep: two u64int bitboards (black_bb, white_bb)
Notes:
    Top-left sq is bit 0
    bit i means row (i//8) and col (i%8)
    0 means no piece, 1 means has piece there

All hot functions are JIT-compiled with Numba
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit, uint64

################################# CONSTANTS #################################

# Turn
BLACK = 1
WHITE = 2
EMPTY = 0

# Starting board
INIT_BLACK = uint64(1 << 28 | 1 << 35)
INIT_WHITE = uint64(1 << 27 | 1 << 36)

# Bounds
# shifting right (West) until reaches edge, prevents looping back to H File
WEST_BOUND = uint64(0xFEFEFEFEFEFEFEFE) # 1111 1110
EAST_BOUND = uint64(0x7F7F7F7F7F7F7F7F) # 0111 1111

############################## HELPER FUNCTIONS #############################

@njit(cache=True)
def lsb(bb: uint64):
    lsb = bb & -bb
    idx = 0
    while (lsb >> idx) > 1:
        idx += 1
    return idx, bb ^ lsb

################################### GAME ####################################

@njit(cache=True)
def move_gen(me: uint64, opp: uint64): # parameters are 64bit bitboards
    empty = uint64(~(me | opp))
    moves = uint64(0)

    # North
    candidates = opp & (me >> 8)
    for _ in range(6):
        candidates |= opp & (candidates >> 8)
    moves |= empty & (candidates >> 8)

    # South
    candidates = opp & (me << 8)
    for _ in range(6):
        candidates |= opp & (candidates << 8)
    moves |= empty & (candidates << 8)

    # East
    candidates = opp & (me << 1) & EAST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates << 1) & EAST_BOUND
    moves |= empty & (candidates << 1) & EAST_BOUND

    # West
    candidates = opp & (me >> 1) & WEST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates >> 1) & WEST_BOUND
    moves |= empty & (candidates >> 1) & WEST_BOUND

    # NE
    candidates = opp & (me >> 7) & EAST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates >> 7) & EAST_BOUND
    moves |= empty & (candidates >> 7) & EAST_BOUND

    # SE
    candidates = opp & (me << 9) & EAST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates << 9) & EAST_BOUND
    moves |= empty & (candidates << 9) & EAST_BOUND

    # SW
    candidates = opp & (me << 7) & WEST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates << 7) & WEST_BOUND
    moves |= empty & (candidates << 7) & WEST_BOUND

    # NW
    candidates = opp & (me >> 9) & WEST_BOUND
    for _ in range(6):
        candidates |= opp & (candidates >> 9) & WEST_BOUND
    moves |= empty & (candidates >> 9) & WEST_BOUND

    return moves

@njit(cache=True)
def apply_move(me: uint64, opp: uint64, move_idx: int):
    move = uint64(1) << move_idx
    flipped = uint64(0)

    # North
    in_line = uint64(0)
    checker = move >> 8
    while checker & opp:
        in_line |= checker
        checker >>= 8
    if (checker & me):
        flipped |= in_line
    
    # South
    in_line = uint64(0)
    checker = move << 8
    while checker & opp:
        in_line |= checker
        checker <<= 8
    if (checker & me):
        flipped |= in_line
    
    # East
    in_line = uint64(0)
    checker = (move << 1) & EAST_BOUND
    while checker & opp:
        in_line |= checker
        checker <<= 1
        checker &= EAST_BOUND
    if (checker & me):
        flipped |= in_line
    
    # West
    in_line = uint64(0)
    checker = (move >> 1) & WEST_BOUND
    while checker & opp:
        in_line |= checker
        checker >>= 1
        checker &= WEST_BOUND
    if (checker & me):
        flipped |= in_line

    # NE
    in_line = uint64(0)
    checker = (move >> 7) & EAST_BOUND
    while checker & opp:
        in_line |= checker
        checker >>= 7
        checker &= EAST_BOUND
    if (checker & me):
        flipped |= in_line

    # SE
    in_line = uint64(0)
    checker = (move << 9) & EAST_BOUND
    while checker & opp:
        in_line |= checker
        checker <<= 9
        checker &= EAST_BOUND
    if (checker & me):
        flipped |= in_line

    # SW
    in_line = uint64(0)
    checker = (move << 7) & WEST_BOUND
    while checker & opp:
        in_line |= checker
        checker <<= 7
        checker &= WEST_BOUND
    if (checker & me):
        flipped |= in_line

    # NW
    in_line = uint64(0)
    checker = (move >> 9) & WEST_BOUND
    while checker & opp:
        in_line |= checker
        checker >>= 9
        checker &= WEST_BOUND
    if (checker & me):
        flipped |= in_line

    me |= (flipped | move)
    opp ^= flipped
    return me, opp

################################ EASE OF USE ################################

@njit(cache=True)
def get_moves(me: uint64, opp: uint64):
    moves_bb = move_gen(me, opp)
    moves = []
    while moves_bb:
        idx, moves_bb = lsb(moves_bb)
        moves.append(idx)
    return moves

# TODO: translate algebraic notation of squares to bitboard index (A4 -> 3)
# A0 is top-left from black perspective