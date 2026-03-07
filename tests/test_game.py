from othello.game import Game
from othello.board import BLACK, WHITE


game = Game()

while not game.game_over:
    game.print_board()
    print(f"Legal moves for {'Black' if game.current_player == BLACK else 'White'}: {game.legal_moves}")
    
    try:
        #Get user input as the move
        move = int(input("Enter move index: "))
        game.make_move(move)

    except ValueError as e:
        print(f"Invalid move, try again: {e}")
