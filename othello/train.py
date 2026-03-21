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


from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
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
    "temperature":          1.0,
    "temperature_drop":     20,

    # get_phase_config() linearly interpolates these over n_iterations,
    # so searches and game count start low (cheap/noisy) and ramp up as the model improves.
    "num_searches_start":  150,
    "num_searches_end":    500,

    "n_games_start":       75,
    "n_games_end":         200,
    
    "n_iterations":        1000,
 
    # Replay buffer
    "buffer_max_games":     2000,

    # Training
    "n_epochs":             8,
    "batch_size":           2048, 

    "lr":                       1e-3,
    "lr_iteration_threshold":   300,
    "lr_divisor":               5,

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

    # Parallelism
    "n_workers":            4,   # worker processes for self-play and evaluation
}

################################# HELPERS #################################

class ReplayBuffer:
    def __init__(self, max_pos):
        self.max_pos  = max_pos
        self.boards   = np.empty((max_pos, 3, 8, 8), dtype=np.float32)
        self.policies = np.empty((max_pos, 64),      dtype=np.float32)
        self.values   = np.empty(max_pos,             dtype=np.float32)
        self.size     = 0
        self.ptr      = 0   # next write position (ring buffer)

    def add_game(self, game_data):
        for board, policy, value in game_data:
            self.boards[self.ptr]   = board
            self.policies[self.ptr] = policy
            self.values[self.ptr]   = value
            self.ptr  = (self.ptr + 1) % self.max_pos
            self.size = min(self.size + 1, self.max_pos)

    def get_batch(self, batch_size):
        indices  = np.random.randint(0, self.size, size=min(batch_size, self.size))
        boards   = torch.from_numpy(self.boards[indices])
        policies = torch.from_numpy(self.policies[indices])
        values   = torch.from_numpy(self.values[indices])
        return boards, policies, values

    def __len__(self):
        return self.size
    
def get_phase_config(base_config, iteration):
    cfg      = base_config.copy()
    progress = min(iteration / base_config["n_iterations"], 1.0)
    cfg["num_searches"]      = int(base_config["num_searches_start"] + (base_config["num_searches_end"] - base_config["num_searches_start"]) * progress)
    cfg["n_self_play_games"] = int(base_config["n_games_start"]      + (base_config["n_games_end"]      - base_config["n_games_start"])      * progress)
    return cfg

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

def process_game(history, winner):
    samples = []
    for board_planes, policy, player, turn in history:
        val = 0 if winner is None else (1 if winner == player else -1)
        if turn > 3:
            for b, p in get_rotations(board_planes, policy):
                samples.append((b, p, val))
    return samples

def _raw_state_dict(model):
    """Return state dict with plain keys, regardless of whether model is torch.compiled."""
    m = getattr(model, '_orig_mod', model)
    return {k: v.cpu() for k, v in m.state_dict().items()}

################################# SELF-PLAY #################################

def _self_play_worker(state_dict, args, device, n_games):
    """Worker process: reconstructs model, plays n_games, returns augmented samples."""
    model = ResNet(args["n_res_blocks"], args["filters"]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    games       = [Game() for _ in range(n_games)]
    histories   = [[] for _ in range(n_games)]
    move_counts = [0] * n_games
    mcts        = MCTS(games[0], model, args, device)
    active      = set(range(n))

    while active:
        active_list = sorted(active)

        # All active games in one batch; pass per-game temperature so greedy
        # games (past temperature_drop) get temperature=0 without a second forward pass.
        temps   = [args["temperature"] if move_counts[i] < args["temperature_drop"] else 0 for i in active_list]
        results = {}
        for idx, (move, policy) in zip(active_list, mcts.search_batch([games[i] for i in active_list], temps)):
            results[idx] = (move, policy)

        for i in active_list:
            move, policy = results[i]
            if move is None:
                if games[i].game_over:
                    active.discard(i)
                continue
            histories[i].append((board_to_planes(games[i]), policy, games[i].current_player, move_counts[i]))
            games[i].make_move(move)
            move_counts[i] += 1
            if games[i].game_over:
                active.discard(i)

    # Parallelize retroactive value labeling + 8x symmetry generation across games.
    # NumPy releases the GIL during rot90/flip, so threads run concurrently.
    total_gen = 0
    with ThreadPoolExecutor as executor:
        for samples in executor.map(process_game,
                                    histories,
                                    [games[i].winner for i in range(n)]):
            buffer.add_game(samples)
            total_gen += len(samples)


def run_self_play(buffer: ReplayBuffer, model: ResNet, args, device):
    n_workers        = args.get("n_workers", 1)
    n_games          = args["n_self_play_games"]
    games_per_worker = n_games // n_workers
    state_dict       = _raw_state_dict(model)
    worker_args      = [(state_dict, args, device, games_per_worker)] * n_workers

    if n_workers == 1:
        all_samples = [_self_play_worker(*worker_args[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            all_samples = pool.starmap(_self_play_worker, worker_args)

    total = 0
    for samples in all_samples:
        buffer.add_game(samples)
        total += len(samples)
    return total

################################# TRAINING #################################

def run_training(
    model: ResNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    args: dict,
    device: str,
):
    model.train()
    value_criterion = nn.MSELoss()

    batch_size        = args["batch_size"]
    batches_per_epoch = max(1, len(buffer) // batch_size)
    total_steps       = args["n_epochs"] * batches_per_epoch
    p_loss_sum = v_loss_sum = 0.0

    t_batch = t_fwdbwd = 0.0

    for _ in range(total_steps):
        t0 = time.time()
        boards, policies, values = buffer.get_batch(batch_size)
        boards   = boards.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values   = values.to(device,   non_blocking=True)
        t_batch += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.float16):
            pred_policy, pred_value = model(boards)
            p_loss = -(policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
            v_loss = value_criterion(pred_value.squeeze(-1), values)
            loss   = p_loss + v_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_fwdbwd += time.time() - t0

        p_loss_sum += p_loss.item()
        v_loss_sum += v_loss.item()

    print(f"  [timing] {total_steps} steps | get_batch: {t_batch:.2f}s ({t_batch/total_steps*1000:.1f}ms/step) | fwd+bwd: {t_fwdbwd:.2f}s ({t_fwdbwd/total_steps*1000:.1f}ms/step)")
    return p_loss_sum / total_steps, v_loss_sum / total_steps

################################# EVALUATION #################################

def _eval_worker(new_state_dict, best_state_dict, args, device, n_games, new_plays_as):
    """Worker process: plays n_games of new vs best, returns wins for new_model."""
    new_model = ResNet(args["n_res_blocks"], args["filters"]).to(device)
    new_model.load_state_dict(new_state_dict)
    new_model.eval()

    best_model = ResNet(args["n_res_blocks"], args["filters"]).to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    eval_args   = {**args, "temperature": 0, "dirichlet_epsilon": 0}
    opening_ply = args.get("eval_opening_ply", 4)

    games       = [Game() for _ in range(n_games)]
    move_counts = [0] * n_games
    new_mcts    = MCTS(games[0], new_model,  eval_args, device)
    best_mcts   = MCTS(games[0], best_model, eval_args, device)

    while True:
        active = [i for i in range(n_games) if not games[i].game_over]
        if not active:
            break

        opening     = [i for i in active if move_counts[i] < opening_ply]
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
    for i in range(n_games):
        winner = games[i].winner
        wins += 1 if winner == new_plays_as[i] else (0.5 if winner is None else 0)
        color = 'BLACK' if new_plays_as[i] == BLACK else 'WHITE'
        label = 'win' if winner == new_plays_as[i] else ('draw' if winner is None else 'loss')
        print(f"  new={color} → {label}")
    return wins


def evaluate_models(new_model: ResNet, best_model: ResNet, args, device) -> float:
    n_workers       = args.get("n_workers", 1)
    n               = args["n_eval_games"]
    total           = n * 2
    new_state_dict  = _raw_state_dict(new_model)
    best_state_dict = _raw_state_dict(best_model)

    # Split: first n games new plays BLACK, next n new plays WHITE.
    # Distribute evenly across workers.
    games_per_worker  = total // n_workers
    all_new_plays_as  = [BLACK] * n + [WHITE] * n
    worker_args = [
        (new_state_dict, best_state_dict, args, device,
         games_per_worker, all_new_plays_as[w * games_per_worker:(w + 1) * games_per_worker])
        for w in range(n_workers)
    ]

    if n_workers == 1:
        total_wins = _eval_worker(*worker_args[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.starmap(_eval_worker, worker_args)
        total_wins = sum(results)

    return total_wins / total


def training_loop(model: ResNet, optimizer: torch.optim.Optimizer, args, device):
    buffer = ReplayBuffer(max_pos=args["buffer_max_games"] * 60 * 8) # 60 turns per game, x8 from position symmetries

    # New checkpoint folder for new training run.
    run_dir  = os.path.join(args["checkpoint_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "loss_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        log_writer.writerow(["iteration", "policy_loss", "value_loss", "total_loss"])

    # Creates a copy of the current model to evaluate against.
    # This allows us to compare the new model's performance against the previous best model without interference from ongoing training updates.
    best_model = ResNet(args["n_res_blocks"], args["filters"]).to(device)
    best_model.load_state_dict(_raw_state_dict(model))
    best_model = torch.compile(best_model)

    # TODO: Add optimizer LR scheduler too.
    for iteration in range(1, args["n_iterations"] + 1):
        phase_args = get_phase_config(args, iteration)

        #Self play phase
        print(f"\n=== Iteration {iteration}/{args['n_iterations']} (searches={phase_args['num_searches']}, games={phase_args['n_self_play_games']}, workers={phase_args['n_workers']}) ===")
        print("Self-play phase:")
        t0 = time.time()
        total_gen = run_self_play(buffer, model, phase_args, device)
        print(f"Generated {total_gen} training samples. Buffer size: {len(buffer)} ({time.time()-t0:.1f}s)")

        #LR decay
        if iteration % args["lr_iteration_threshold"] == 0:
            for pg in optimizer.param_groups:
                pg["lr"] /= args["lr_divisor"]
            print(f"LR decayed to {optimizer.param_groups[0]['lr']:.2e}")

        # Training phase, train `model`
        print("Training phase:")
        t0 = time.time()
        p_loss, v_loss = run_training(model, optimizer, buffer, args, device)
        print(f"Policy loss: {p_loss:.4f}, Value loss: {v_loss:.4f}, Total: {p_loss+v_loss:.4f} ({time.time()-t0:.1f}s)")
        log_writer.writerow([iteration, f"{p_loss:.6f}", f"{v_loss:.6f}", f"{p_loss+v_loss:.6f}"])
        log_file.flush()

        # Evaluate phase: compare the two models (skip first specified iterations)
        if iteration > 20:
            print("Evaluation phase:")
            t0 = time.time()
            win_rate = evaluate_models(model, best_model, phase_args, device)
            print(f"Win rate: {win_rate:.2%} (threshold: {args['win_rate_threshold']:.2%}) ({time.time()-t0:.1f}s)")

            if win_rate >= args["win_rate_threshold"]:
                print("New model promoted!")
                best_model.load_state_dict(model.state_dict())
            else:
                print("Not promoted, reverting to best model.")
                model.load_state_dict(best_model.state_dict())
        else:
            print(f"Skipping evaluation (iteration {iteration}/25 warmup)")

        # Create a checkpoint every N iterations depending on args["checkpoint_every"]
        if iteration % args["checkpoint_every"] == 0:
            os.makedirs(run_dir, exist_ok=True)
            path = os.path.join(run_dir, f"model_iter{iteration}.pt")
            torch.save(_raw_state_dict(model), path)
            print(f"Checkpoint saved: {path}")

if __name__ == "__main__":
    RESUME_CHECKPOINT = "checkpoints/20260320_140904/model_iter5.pt"

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("[WARNING]: Using CPU")
    
    model = ResNet(CONFIG["n_res_blocks"], CONFIG["filters"]).to(device)

    if device == "cuda": model = torch.compile(model, backend="cudagraphs")

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
        print(f"Resumed from {RESUME_CHECKPOINT}")

    # TODO: Add Optimizer State Checkpointing
    # TODO: Should we switch to AdamW?
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    training_loop(model, optimizer, CONFIG, device)