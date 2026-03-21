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
from datetime import datetime
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from othello.board import BLACK, WHITE
from othello.game import Game
from othello.model import board_to_planes, ResNet
from othello.BetaFish import MCTS

################################# CONSTANTS #################################

CONFIG = {
    # Architecture
    "n_res_blocks":         10,
    "filters":              128,
 
    # Self-play
    "n_iterations":         1000,
    "n_self_play_games":    100,       # 25 → 50: more diverse positions per iter
    "num_searches":         150,     
    "temperature":          1.0,
    "temperature_drop":     20,
 
    # Replay buffer
    "buffer_max_games":     2000,
 
    # Training
    "n_epochs":             100,
    "batch_size":           256,      # 128 → 256: better GPU utilisation
    "lr":                   1e-3,
    "weight_decay":         1e-4,
 
    # Evaluation
    "n_eval_games":         25,    
    "win_rate_threshold":   0.52,    
 
    # MCTS
    "dirichlet_alpha":      0.3,
    "dirichlet_epsilon":    0.25,
    "exploration_constant": 1.5,
 
    # Checkpointing
    "checkpoint_dir":       "checkpoints",
    "checkpoint_every":     5,
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

def run_self_play(buffer: ReplayBuffer, model: ResNet, args, device):
    # Runs all n_self_play_games simultaneously, calling search_batch each step
    # so every move across all games shares a single batched NN forward pass.
    #
    # Temperature schedule per game:
    #     move_count < temperature_drop  → args["temperature"]  (explore)
    #     move_count >= temperature_drop → 0                    (greedy)
    model.eval()
    n = args["n_self_play_games"]

    games       = [Game() for _ in range(n)]
    histories   = [[] for _ in range(n)]
    move_counts = [0] * n
    mcts        = MCTS(games[0], model, args, device)

    while True:
        active = [i for i in range(n) if not games[i].game_over]
        if not active:
            break

        # All active games in one batch; pass per-game temperature so greedy
        # games (past temperature_drop) get temperature=0 without a second forward pass.
        temps   = [args["temperature"] if move_counts[i] < args["temperature_drop"] else 0 for i in active]
        results = {}
        for idx, (move, policy) in zip(active, mcts.search_batch([games[i] for i in active], temps)):
            results[idx] = (move, policy)

        for i in active:
            move, policy = results[i]
            if move is None:
                continue
            histories[i].append((board_to_planes(games[i]), policy, games[i].current_player, move_counts[i]))
            games[i].make_move(move)
            move_counts[i] += 1

    #number of training examples
    total_gen = 0
    for i in range(n):
        winner = games[i].winner
        samples = []
        for board_planes, policy, player, turn in histories[i]:
            val = 0 if winner == None else (1 if winner == player else -1)
            if turn > 3:
                for b, p in get_rotations(board_planes, policy):
                    samples.append((b, p, val))
        buffer.add_game(samples)
        total_gen += len(samples)

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
 
    p_loss_sum = v_loss_sum = 0.0

    for _ in range(args["n_epochs"]):
        boards, policies, values = buffer.get_batch(args["batch_size"])
        boards   = boards.to(device)
        policies = policies.to(device)
        values   = values.to(device)

        pred_policy, pred_value = model(boards)

        # Cross-entropy manually (policy head outputs softmax, not logits)
        # Small epsilon prevents log(0)
        p_loss = -(policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
        v_loss = value_criterion(pred_value.squeeze(-1), values)
        loss   = p_loss + v_loss

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients in deep ResNets.
        # max_norm=1.0 is a safe standard value — rarely triggers but catches spikes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        p_loss_sum += p_loss.item()
        v_loss_sum += v_loss.item()

    n = args["n_epochs"]
    return p_loss_sum / n, v_loss_sum / n

################################# EVALUATION #################################

def evaluate_models(new_model: ResNet, best_model: ResNet, args, device) -> float:
    # Runs all 2*n_eval_games simultaneously using search_batch.
    # First n games: new_model plays BLACK. Next n: new_model plays WHITE.
    # Returns new_model's win rate; draws count as 0.5.
    new_model.eval()
    best_model.eval()

    n     = args["n_eval_games"]
    total = n * 2
    eval_args    = {**args, "temperature": 0, "dirichlet_epsilon": 0}
    opening_ply  = args.get("eval_opening_ply", 4)  # random moves before MCTS takes over

    games        = [Game() for _ in range(total)]
    new_plays_as = [BLACK] * n + [WHITE] * n
    move_counts  = [0] * total

    new_mcts  = MCTS(games[0], new_model,  eval_args, device)
    best_mcts = MCTS(games[0], best_model, eval_args, device)

    while True:
        active = [i for i in range(total) if not games[i].game_over]
        if not active:
            break

        # Opening phase: play random moves to diversify starting positions
        opening = [i for i in active if move_counts[i] < opening_ply]
        mcts_active = [i for i in active if move_counts[i] >= opening_ply]

        for i in opening:
            move = random.choice(games[i].legal_moves)
            games[i].make_move(move)
            move_counts[i] += 1

        new_turn  = [i for i in mcts_active if games[i].current_player == new_plays_as[i]]
        best_turn = [i for i in mcts_active if games[i].current_player != new_plays_as[i]]

        results = {}
        if new_turn:
            for idx, (move, _) in zip(new_turn, new_mcts.search_batch([games[i] for i in new_turn])):
                results[idx] = move
        if best_turn:
            for idx, (move, _) in zip(best_turn, best_mcts.search_batch([games[i] for i in best_turn])):
                results[idx] = move

        for i in mcts_active:
            move = results.get(i)
            if move is None:
                continue
            games[i].make_move(move)
            move_counts[i] += 1

    wins = 0.0
    for i in range(total):
        winner = games[i].winner
        wins += 1 if winner == new_plays_as[i] else (0.5 if winner is None else 0)
        color  = 'BLACK' if new_plays_as[i] == BLACK else 'WHITE'
        label  = 'win' if winner == new_plays_as[i] else ('draw' if winner is None else 'loss')
        print(f"  [{i+1}/{total}] new={color} → {label}")

    return wins / total


def training_loop(model: ResNet, optimizer: torch.optim.Optimizer, args, device):
    buffer = ReplayBuffer(max_pos=args["buffer_max_games"] * 60 * 8)

    run_dir  = os.path.join(args["checkpoint_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "loss_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        log_writer.writerow(["iteration", "policy_loss", "value_loss", "total_loss"])

    #Creates a copy of the current model to evaluate against. This allows us to compare the new model's performance against the previous best model without interference from ongoing training updates.
    best_model = ResNet(args["n_res_blocks"], args["filters"]).to(device)
    best_model.load_state_dict(model.state_dict())

    for iteration in range(1, args["n_iterations"] + 1):
        if iteration <= 25:
            args["num_searches"]      = 150
            args["n_self_play_games"] = 100
        elif iteration <= 150:
            args["num_searches"]      = 250
            args["n_self_play_games"] = 150
        else:
            args["num_searches"]      = 400
            args["n_self_play_games"] = 200

        #Self play phase
        print(f"\n=== Iteration {iteration}/{args['n_iterations']} ===")
        print("Self-play phase:")
        total_gen = run_self_play(buffer, model, args, device)
        print(f"Generated {total_gen} training samples. Buffer size: {len(buffer)}")

        #Training phase, train `model`
        print("Training phase:")
        p_loss, v_loss = run_training(model, optimizer, buffer, args, device)
        print(f"Policy loss: {p_loss:.4f}, Value loss: {v_loss:.4f}, Total: {p_loss+v_loss:.4f}")
        log_writer.writerow([iteration, f"{p_loss:.6f}", f"{v_loss:.6f}", f"{p_loss+v_loss:.6f}"])
        log_file.flush()

        #Evaluate phase: compare the two models (skip first specified iterations)
        if iteration > 20:
            print("Evaluation phase:")
            win_rate = evaluate_models(model, best_model, args, device)
            print(f"Win rate: {win_rate:.2%} (threshold: {args['win_rate_threshold']:.2%})")

            if win_rate >= args["win_rate_threshold"]:
                print("New model promoted!")
                best_model.load_state_dict(model.state_dict())
            else:
                print("Not promoted, reverting to best model.")
                model.load_state_dict(best_model.state_dict())
        else:
            print(f"Skipping evaluation (iteration {iteration}/25 warmup)")

        #Create a checkpoint every N iterations depending on args["checkpoint_every"]
        if iteration % args["checkpoint_every"] == 0:
            os.makedirs(run_dir, exist_ok=True)
            path = os.path.join(run_dir, f"model_iter{iteration}.pt")
            torch.save(model.state_dict(), path)
            print(f"Checkpoint saved: {path}")

if __name__ == "__main__":
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    RESUME_CHECKPOINT = "checkpoints/20260320_140904/model_iter5.pt"
    
    model = ResNet(CONFIG["n_res_blocks"], CONFIG["filters"]).to(device)
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
        print(f"Resumed from {RESUME_CHECKPOINT}")

    # model = ResNet(CONFIG["n_res_blocks"], CONFIG["filters"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    training_loop(model, optimizer, CONFIG, device)