# 1. Self-play       — run MCTS games, collect (board, policy, value) samples
# 2. Replay buffer   — store samples, sample random batches for training
# 3. Training step   — compute policy loss + value loss, backprop, update weights
# 4. Training loop   — alternate self-play → train → evaluate → repeat
# 5. Checkpointing   — save model weights periodically
# 6. Parallelize     - for distributed Modal
# 1. done
# 2. Done
# 3. done
# 4. TODO!

# policy loss = KLDivLoss(predicted_policy, mcts_visit_counts)
# value loss  = MSELoss(predicted_value, actual_game_outcome)
# total loss  = policy_loss + value_loss  (equal weighting to start)

# games: [(board (3, 8, 8), policy, value)]

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
from board import BLACK, WHITE
from game import Game
from model import board_to_planes, board_to_tensor, ResNet
from BetaFish import MCTS

################################# CONSTANTS #################################

CONFIG = {
    # Architecture
    "n_res_blocks":         10,
    "filters":              128,
 
    # Self-play
    "n_iterations":         50,     # outer loop: self-play → train → evaluate
    "n_self_play_games":    25,     # games to play per iteration
    "num_searches":         200,    # MCTS simulations per move
    "temperature":          1.0,    # exploration temp (moves 1 to temperature_drop)
    "temperature_drop":     30,     # move number to switch to greedy (temp=0)
 
    # Replay buffer
    # Capacity = buffer_max_games * avg_moves_per_game * 8 symmetries
    # 500 games * 60 moves * 8 = 240,000 positions
    "buffer_max_games":     500,
 
    # Training
    "n_epochs":             4,      # gradient passes over sampled batches per iteration
    "batch_size":           128,
    "lr":                   1e-3,
    "weight_decay":         1e-4,   # L2 regularisation
 
    # Evaluation
    "n_eval_games":         20,     # games per colour side (40 total)
    "win_rate_threshold":   0.55,   # must beat old model by this margin to promote
 
    # MCTS
    "dirichlet_alpha":      0.3,
    "dirichlet_epsilon":    0.25,
    "exploration_constant": 1.5,
 
    # Checkpointing
    "checkpoint_dir":       "checkpoints",
    "checkpoint_every":     5,      # save every N iterations
}

################################# HELPERS #################################

class ReplayBuffer:
    def __init__(self, max_pos):
        self.buffer = deque(maxlen=max_pos)
    
    def add_game(self, game_data):
        self.buffer.extend(game_data)

    def get_batch(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        boards, policies, values = zip(*batch)
        boards = torch.tensor(np.array(boards), dtype=torch.float32)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32)
        return boards, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
# Return all 8 symmetries of board and policy
# Returns 8 x (board (3,8,8), policy (64,))
def get_rotations(board: np.ndarray, policy: np.ndarray):
    policy2d = policy.reshape(8, 8)
    rotations = []
    for i in range(4):
        b = np.rot90(board, i, axes=(1, 2))
        p = np.rot90(policy2d, i)
        rotations.append((b, p.flatten()))
        
        b_mirror = np.flip(b, axis=2)
        p_mirror = np.fliplr(p)
        rotations.append((b_mirror, p_mirror.flatten()))

    return rotations

################################# SELF-PLAY #################################

def play_game(model: ResNet, args, device):
    # Temperature schedule:
    #     Moves < temperature_drop  → args["temperature"]  (explore)
    #     Moves >= temperature_drop → 0                    (greedy)

    # Value labelling:
    #     After the game, every stored position is labelled:
    #        +1.0  if the player to move at that position won
    #        -1.0  if they lost
    #         0.0  if draw
    game = Game()
    bot = MCTS(game, model, args, device)
    history = []
    move_count = 0

    while not game.game_over:
        temp = args["temperature"] if move_count < args["temperature_drop"] else 0
 
        # Temporarily patch temperature for this search call
        saved_temp = args["temperature"]
        args["temperature"] = temp
        move, policy = bot.search()
        args["temperature"] = saved_temp
 
        history.append((board_to_planes(game), policy, game.current_player, move_count))
 
        if move is None:
            print("Error: move was none")
            break

        game.make_move(move)
        move_count += 1
 
    winner = game.winner
    samples = []

    for board_planes, policy, player, turn in history:
        if winner == 0:
            val = 0
        else:
            val = 1 if winner == player else -1
        
        if turn > 3:
            for b, p in get_rotations(board_planes, policy):
                samples.append((b, p, val))
    
    return samples

def run_self_play(buffer: ReplayBuffer, model: ResNet, args, device):
    model.eval()
    total_gen = 0

    for i in range(args["n_self_play_games"]):
        samples = play_game(model, args, device)
        buffer.add_game(samples)
        total_gen += len(samples)
        print(f"  Game {i+1}/{n} — {len(samples)//8} moves, ")
    
    return total_gen

################################# TRAINING #################################

def run_training(
    model: ResNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    args: dict,
    device: str
):
    model.train()
    value_criterion = nn.MSELoss()
 
    p_loss_out = v_loss_out = 0.0
 
    for epoch in range(args["n_epochs"]):
        boards, policies, values = buffer.get_batch(args["batch_size"])
        boards   = boards.to(device)
        policies = policies.to(device)
        values   = values.to(device)
 
        pred_policy, pred_value = model(boards)
 
        # Cross-entropy manually (policy head outputs softmax, not logits)
        # Small epsilon prevents log(0)
        p_loss = -(policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
        v_loss = value_criterion(pred_value, values)
        loss   = p_loss + v_loss
 
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients in deep ResNets.
        # max_norm=1.0 is a safe standard value — rarely triggers but catches spikes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 
        p_loss_out = p_loss.item()
        v_loss_out = v_loss.item()
 
    return p_loss_out, v_loss_out
