from .board import BLACK, WHITE, apply_move, move_gen, get_moves, popcount, get_value_and_terminated
import math
import random
import numpy as np

# MCTS Bot for Othello
# ----------------------------------------
# Monte Carlo Tree Search works by repeatedly simulating random games
# from the current position and using the results to guide the search.
#
# Each iteration does 4 steps:
#   1. Selection   - walk the tree using PUCT to pick a promising node
#   2. Expansion   - add a new child node for an unexplored move
#   3. Simulation  - play out a random game from that node to the end/Use nn to determine value
#   4. Backprop    - update win/visit counts up the tree
#
# After N iterations, return the move with the most visits.


class Node:
    def __init__(self, game_state, args, parent=None, action_taken=None, prior=0):
        #Unpack the tuple
        black, white, player = game_state
        self.black_bb = np.uint64(black)
        self.white_bb = np.uint64(white)
        self.current_player = int(player)

        self.args = args #a dictionary of hyperparameters to configure search. Ex. num_searches, exploration_constant...

        self.parent = parent #the parents node
        self.action_taken = action_taken #move that led to this state

        self.prior = prior #probability from parent
        self.value_sum = 0
        self.visit_count = 0

        self.children = []          # expanded child nodes
        if self.current_player == BLACK:
            self.expandable_moves = get_moves(self.black_bb, self.white_bb)
        else:
            self.expandable_moves = get_moves(self.white_bb, self.black_bb)


    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum/self.visit_count
        
    @property
    def is_expanded(self):
        return len(self.children) > 0 and len(self.expandable_moves) == 0
    
    @property
    def is_terminal(self):
        return (move_gen(self.black_bb, self.white_bb) == 0 and
                move_gen(self.white_bb, self.black_bb) == 0)
    
    def get_puct(self, child):
        if child.visit_count == 0:
            exploitation = 0
        else:
            exploitation = child.value_sum/child.visit_count

        exploration = self.args["exploration_constant"]*child.prior* (math.sqrt(self.visit_count)/(1 + child.visit_count))

        return exploitation + exploration

    #Finds the child with the best PUCT score
    def select_child(self):
        best_boy = None
        best_puct = -math.inf

        for child in self.children:
            current_puct = self.get_puct(child)

            if current_puct > best_puct:
                best_puct = current_puct
                best_boy = child

        return best_boy
    
    def expand(self):
        #Pick a random legal move
        idx = random.randrange(len(self.expandable_moves))
        move = self.expandable_moves.pop(idx)

        #make the move
        if self.current_player == BLACK:
            new_black, new_white = apply_move(self.black_bb, self.white_bb, move)

            #Make sure white has legal moves, otherwise its blacks turn again
            if move_gen(new_white, new_black) == 0:
                player_turn = BLACK
            else:
                player_turn = WHITE
        else:
            new_white, new_black = apply_move(self.white_bb, self.black_bb, move)

            if move_gen(new_black, new_white) == 0:
                player_turn = WHITE
            else:
                player_turn = BLACK


        #def __init__(self, game_state, args, parent=None, action_taken=None, prior=0):
        child = Node((new_black, new_white, player_turn), self.args, self, move, 0) #NO PRIOR PROBABILITIES FROM NETWORK YET

        self.children.append(child)

        return child
    

    def simulate(self):
        return 0

    def simulate_rollout(self):
        black_bb, white_bb, current_player = self.black_bb, self.white_bb, self.current_player

        while True:
            black_moves = get_moves(black_bb, white_bb)
            white_moves = get_moves(white_bb, black_bb)
            if not black_moves and not white_moves:
                break
            if current_player == BLACK:
                if black_moves:
                    move = random.choice(black_moves)
                    black_bb, white_bb = apply_move(black_bb, white_bb, move)
                    black_bb, white_bb = np.uint64(black_bb), np.uint64(white_bb)
                current_player = WHITE
            else:
                if white_moves:
                    move = random.choice(white_moves)
                    white_bb, black_bb = apply_move(white_bb, black_bb, move)
                    white_bb, black_bb = np.uint64(white_bb), np.uint64(black_bb)
                current_player = BLACK

        black_count = popcount(black_bb)
        white_count = popcount(white_bb)
        if black_count > white_count:
            winner = BLACK
        elif white_count > black_count:
            winner = WHITE
        else:
            return 0

        return 1 if winner == self.current_player else -1


    def most_visited_child(self):
        best_boy = None
        highest_visits = 0

        for child in self.children:
            if child.visit_count > highest_visits:
                best_boy = child
                highest_visits = child.visit_count

        return best_boy #Returns None if there are no children

    def backpropagate_iterative(self, value):
        node = self

        #when there is still a parent node
        while node is not None:

            #these are the 2 values we update
            node.value_sum += value
            node.visit_count += 1

            #Go up the tree to the parent node, negate value each iteration up
            node = node.parent
            value = -value
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        #Recursive call, if there is a parent, keep traversing upwards until you get to the root
        if self.parent:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def add_dirichlet_noise(self, node):
        num_moves = len(node.children)

        #Created a list of num_moves, filled with alpha
        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * num_moves)

        epsilon = self.args['dirichlet_epsilon']
        
        for i, child in enumerate(node.children):
            #Computes a weighted average for the actual child.prior, 
            # and the random noice, with epsilon weight on the noise
            child.prior = (1-epsilon)*child.prior + epsilon*noise[i]
            
    def choose_move(self, root, temperature):
        #0 temperature is no exploration - always pick best move
        #1 and above are more explorative
        visit_counts = [child.visit_count for child in root.children]
        moves = [child.action_taken for child in root.children]

        #Greedy - Always picks move visits
        if temperature == 0:
            max_visit_count = max(visit_counts)
            max_idx = visit_counts.index(max_visit_count) 
            return moves[max_idx]
        else:
            # sample proportional to visit_count^(1/temperature)
            counts = [v ** (1/temperature) for v in visit_counts]
            total = sum(counts)

            #Convert the visit counts to a list of probabilities to use for choosing move
            probabilities = [v/total for v in counts]
            return random.choices(moves, weights=probabilities)[0]
    
    def search(self):
        root = Node((self.game.black_bb, self.game.white_bb, self.game.current_player), self.args)

        for i in range(self.args['num_searches']):
            node = root

            #1. SELECTION
            #if the node is fully expanded, go downwards and select its children
            while not node.is_terminal and node.is_expanded:
                node = node.select_child()
            
            #Check if the game is over (terminal) and get the value
            value, is_terminal = get_value_and_terminated(node.black_bb, node.white_bb, node.current_player)
            
            #On a unexplored, non-terminal leaf node
            if not is_terminal:

                # Must-pass case, no expandable moves but its not a terminal state
                if len(node.expandable_moves) == 0:
                    next_player = WHITE if node.current_player == BLACK else BLACK

                    pass_child = Node(
                        (node.black_bb, node.white_bb, next_player),
                        node.args,
                        parent=node,
                        action_taken=-1,   # pass
                        prior=0
                    )

                    node.children.append(pass_child)
                    node = pass_child

                else:
                    # 2. Expansion
                    node = node.expand()
 

                #3. SIMULATION
                value = node.simulate_rollout()


            #Once we reach either a simulated end, or actual terminal node, backpropagate the value
            node.backpropagate(value)

        #Now after doing args.['num_searches'], we choose a move
        return self.choose_move(root, self.args['temperature'])
        

            
                

    




