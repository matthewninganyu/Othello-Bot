import torch
from othello.game import Game
from othello.board import BLACK, WHITE, notation_to_idx, idx_to_notation
from othello.model import ResNet
from othello.BetaFish import MCTS
import os

CHECKPOINT = "checkpoints/20260319_000252/model_iter324.pt"  # set to None to use untrained model

args = {
    "num_searches":         200,
    "exploration_constant": 1.5,
    "temperature":          0,
    "dirichlet_alpha":      0,
    "dirichlet_epsilon":    0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(n_res_blocks=10, filters=128).to(device)

if CHECKPOINT and os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded model from {CHECKPOINT}")
else:
    print("No checkpoint found — using untrained model")

model.eval()

game = Game()
mcts = MCTS(game, model, args, device)

while not game.game_over:
    game.print_board()
    current = 'Black' if game.current_player == BLACK else 'White'
    print(f"Legal moves for {current}: {[idx_to_notation(m) for m in game.legal_moves]}")

    if game.current_player == BLACK:
        # Human plays Black
        while True:
            try:
                move = str(input("Enter move notation: "))
                move_index = notation_to_idx(move)
                game.make_move(move_index)
                break
            except ValueError as e:
                print(f"Invalid move, try again: {e}")
    else:
        # Bot plays White
        move_index, _ = mcts.search()
        print(f"Bot plays: {idx_to_notation(move_index)}")
        game.make_move(move_index)

game.print_board()
winner = game.winner
print(f"Game over! Winner: {'Black' if winner == BLACK else 'White' if winner == WHITE else 'Draw'}")
