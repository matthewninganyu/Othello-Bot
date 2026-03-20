import random
import torch
from othello.game import Game
from othello.board import BLACK, WHITE
from othello.model import ResNet
from othello.BetaFish import MCTS

# ── Config ────────────────────────────────────────────────────────────────────

#Select the two checkpoints to compare. Set to None to use untrained model.
CHECKPOINT_A = "checkpoints/20260319_213435/model_iter100.pt"
CHECKPOINT_B = "checkpoints/20260319_213435/model_iter20.pt"

N_GAMES       = 100    # total games (split evenly between A-as-black and A-as-white)
NUM_SEARCHES  = 200
OPENING_PLY   = 4     # random opening moves before MCTS takes over (diversity)

ARGS = {
    "num_searches":         NUM_SEARCHES,
    "exploration_constant": 1.5,
    "temperature":          0,
    "dirichlet_alpha":      0.3,
    "dirichlet_epsilon":    0.0,   # no noise during evaluation
}

# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    m = ResNet(n_res_blocks=10, filters=128).to(device)
    if path:
        m.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {path}")
    else:
        print("Using untrained model")
    m.eval()
    return m

model_a = load_model(CHECKPOINT_A)
model_b = load_model(CHECKPOINT_B)

mcts_a = MCTS(None, model_a, ARGS, device)
mcts_b = MCTS(None, model_b, ARGS, device)

# ── Run games ─────────────────────────────────────────────────────────────────
# First N//2 games: A plays BLACK. Next N//2: A plays WHITE.
n      = N_GAMES // 2
total  = n * 2

games        = [Game() for _ in range(total)]
a_plays_as   = [BLACK] * n + [WHITE] * n
move_counts  = [0] * total

print(f"\nRunning {total} games ({n} as BLACK, {n} as WHITE) …\n")

while True:
    active = [i for i in range(total) if not games[i].game_over]
    if not active:
        break

    # Random opening moves
    opening    = [i for i in active if move_counts[i] < OPENING_PLY]
    mcts_phase = [i for i in active if move_counts[i] >= OPENING_PLY]

    for i in opening:
        move = random.choice(games[i].legal_moves)
        games[i].make_move(move)
        move_counts[i] += 1

    a_turn = [i for i in mcts_phase if games[i].current_player == a_plays_as[i]]
    b_turn = [i for i in mcts_phase if games[i].current_player != a_plays_as[i]]

    results = {}
    if a_turn:
        for idx, (move, _) in zip(a_turn, mcts_a.search_batch([games[i] for i in a_turn])):
            results[idx] = move
    if b_turn:
        for idx, (move, _) in zip(b_turn, mcts_b.search_batch([games[i] for i in b_turn])):
            results[idx] = move

    for i in mcts_phase:
        move = results.get(i)
        if move is None:
            continue
        games[i].make_move(move)
        move_counts[i] += 1

# ── Results ───────────────────────────────────────────────────────────────────
a_wins = 0.0
for i in range(total):
    winner = games[i].winner  # BLACK, WHITE, or None (draw)
    color  = 'BLACK' if a_plays_as[i] == BLACK else 'WHITE'
    if winner is None:
        result = 'Draw';         a_wins += 0.5
    elif winner == a_plays_as[i]:
        result = 'Model A wins'; a_wins += 1
    else:
        result = 'Model B wins'
    print(f"  [{i+1:>2}/{total}] A={color}  →  {result}")

print(f"\nModel A win rate: {a_wins}/{total} = {a_wins/total:.1%}")
print(f"Model B win rate: {total-a_wins}/{total} = {(total-a_wins)/total:.1%}")
