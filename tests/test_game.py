from othello.game import Game
from othello.board import BLACK, WHITE, notation_to_idx, idx_to_notation


game = Game()

while not game.game_over:
    game.print_board()
    print(f"Legal moves for {'Black' if game.current_player == BLACK else 'White'}: {[idx_to_notation(m) for m in game.legal_moves]}")
    
    try:
        #Get user input as the move
        move = str(input("Enter move notation: "))
        move_index = notation_to_idx(move)

        game.make_move(move_index)

    except ValueError as e:
        print(f"Invalid move, try again: {e}")
