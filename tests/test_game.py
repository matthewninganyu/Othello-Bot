from othello.game import Game
from othello.board import BLACK, WHITE, notation_to_idx, idx_to_notation
from othello.BetaFish import MCTS

args = {"num_searches": 800, "exploration_constant": 1.41, "temperature": 0}

game = Game()
mcts = MCTS(game, args)

while not game.game_over:
    game.print_board()
    current = 'Black' if game.current_player == BLACK else 'White'
    print(f"Legal moves for {current}: {[idx_to_notation(m) for m in game.legal_moves]}")

    if game.current_player == BLACK:
        # Human plays Black
        try:
            move = str(input("Enter move notation: "))
            move_index = notation_to_idx(move)
            game.make_move(move_index)
        except ValueError as e:
            print(f"Invalid move, try again: {e}")
    else:
        # Bot plays White
        move_index = mcts.search()
        print(f"Bot plays: {idx_to_notation(move_index)}")
        game.make_move(move_index)

game.print_board()
winner = game.winner
print(f"Game over! Winner: {'Black' if winner == BLACK else 'White' if winner == WHITE else 'Draw'}")
