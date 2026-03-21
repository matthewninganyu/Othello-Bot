from othello.board import BLACK, WHITE, apply_move, move_gen, get_moves, popcount, get_value_and_terminated
import math
import random
import numpy as np
from othello.model import ResNet

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

################################# CONSTANTS #################################


################################# OBJECTS #################################
# args keys:
#   'num_searches'         (int)   - MCTS iterations per move
#   'exploration_constant' (float) - PUCT exploration weight; higher = more exploration
#   'dirichlet_alpha'      (float) - Dirichlet noise concentration (~0.3 for Othello)
#   'dirichlet_epsilon'    (float) - Noise mix at root; 0 = pure policy, 1 = pure noise
#   'temperature'          (float) - Move sampling; 0 = greedy, 1 = visit-proportional

class Node:
    # Attributes:
    #   black_bb, white_bb - position bitboards
    #   current_player     - turn (BLACK or WHITE)
    #   args               - dictionary of hyperparameters to configure search
    #   parent             - parent node (None for root)
    #   action_taken       - move index that led to this node
    #   prior              - probability from parent
    #   value_sum          - total value accumulated across visits
    #   visit_count        - number of visits
    #   children           - dict {move_idx: Node}
    
    #   value              - (int ) average of value_sum
    #   is_expanded        - (bool) reached fully expanded state
    #   is_terminal        - (bool) is in game end state
    
    def __init__(self, game_state, args, parent=None, action_taken=None, prior=0):        
        # Parameters
        # - game_state: tuple (black_bitboard, white_bitboard, current_player)
        # - args: search hyperparameters (dict)
        # - parent: parent Node or None for root
        # - action_taken: move index that led to this node (or -1 for pass)
        # - prior: prior probability from policy (float)

        # Unpack the tuple
        black, white, player = game_state
        self.black_bb = np.uint64(black)
        self.white_bb = np.uint64(white)
        self.current_player = int(player)

        self.args = args # dictionary of hyperparameters, e.g. num_searches, exploration_constant

        self.parent = parent
        self.action_taken = action_taken

        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0

        # Child nodes and remaining legal moves to expand
        self.children = []
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
        # Compute the PUCT score for a child node.
        # Returns exploitation + exploration term used to balance search.
        
        if child.visit_count == 0:
            exploitation = 0
        else:
            # Negate: child.value_sum is from the child's player's perspective,
            # but the parent wants to maximise from its own perspective.
            exploitation = -(child.value_sum / child.visit_count)

        exploration = (
            self.args["exploration_constant"]
            * child.prior
            * (math.sqrt(self.visit_count) / (1 + child.visit_count))
        )

        return exploitation + exploration

    # Finds the child with the best PUCT score
    def select_child(self):
        best_boy = None
        best_puct = -math.inf

        for child in self.children:
            current_puct = self.get_puct(child)

            if current_puct > best_puct:
                best_puct = current_puct
                best_boy = child

        return best_boy
    


    def expand_random(self):
        # Pick a random legal move
        idx = random.randrange(len(self.expandable_moves))
        move = self.expandable_moves.pop(idx)

        # Make the move and determine next player's turn
        if self.current_player == BLACK:
            new_black, new_white = apply_move(self.black_bb, self.white_bb, move)
            # Ensure the opponent has moves; if not, turn remains the same
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

        # Determine winner from final piece counts
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

        return best_boy # Returns None if there are no children

    def backpropagate(self, value):
        node = self

        # When there is still a parent node
        while node is not None:

            # Update accumulated value and visit count
            node.value_sum += value
            node.visit_count += 1

            # Move up the tree and flip the perspective of the value
            node = node.parent
            value = -value
    

class MCTS:
    def __init__(self, game, model, args, device="cuda"):
        self.game = game
        self.model = model
        self.args = args 
        #{"num_searches": #, "exploration_constant": #, "temperature": #, dirichlet_alpha: #, dirichlet_epsilon: #}
        self.device = device

    def eval(self, node):
        policy_probs, value = self.model.inference(node.black_bb, node.white_bb, node.current_player, device=self.device)
        return policy_probs, value
    
    def _attach_children(self, node: Node, policy_probs):
        # Shared helper: create child nodes from policy probs and mark node as expanded
        legal_moves = node.expandable_moves

        priors = np.zeros(64, dtype=np.float64)
        for m in legal_moves:
            priors[m] = float(policy_probs[m])

        s = priors.sum()
        if s <= 0:
            for m in legal_moves:
                priors[m] = 1.0 / len(legal_moves)
        else:
            priors /= s

        for m in legal_moves:
            if node.current_player == BLACK:
                new_black, new_white = apply_move(node.black_bb, node.white_bb, m)
                next_player = BLACK if move_gen(new_white, new_black) == 0 else WHITE
            else:
                new_white, new_black = apply_move(node.white_bb, node.black_bb, m)
                next_player = WHITE if move_gen(new_black, new_white) == 0 else BLACK

            child = Node((new_black, new_white, next_player), node.args, parent=node, action_taken=m, prior=float(priors[m]))
            node.children.append(child)

        node.expandable_moves = []

    def expand_node(self, node: Node):
        policy_probs, value = self.eval(node)

        if len(node.expandable_moves) == 0:
            return value

        self._attach_children(node, policy_probs)
        return value


    def add_dirichlet_noise(self, node):
        if self.args["dirichlet_epsilon"] == 0:
            return  # skip noise entirely during evaluation/testing
        
        num_moves = len(node.children)

        #Created a list of num_moves, filled with alpha
        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * num_moves)

        epsilon = self.args['dirichlet_epsilon']

        for i, child in enumerate(node.children):
            # Computes a weighted average for the actual child.prior, 
            # and the random noice, with epsilon weight on the noise
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
            
    def choose_move(self, root, temperature):
        #0 temperature is no exploration - always pick best move
        #1 and above are more explorative
        valid = [c for c in root.children if c.action_taken >= 0]
        if not valid:
            return None
        visit_counts = [child.visit_count for child in valid]
        moves = [child.action_taken for child in valid]

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

        # Expand root with network priors before starting iterations
        self.expand_node(root)
        # Add exploration noise at the root (AlphaZero style)
        self.add_dirichlet_noise(root)

        for i in range(self.args['num_searches']):
            node = root

            # 1. SELECTION
            # If the node is fully expanded, go downwards and select its children
            while not node.is_terminal and node.is_expanded:
                node = node.select_child()
            
            # Check if the game is over (terminal) and get the value
            value, is_terminal = get_value_and_terminated(node.black_bb, node.white_bb, node.current_player)

            # On a unexplored, non-terminal leaf node
            if not is_terminal:

                # Must-pass case, no expandable moves but its not a terminal state
                if len(node.expandable_moves) == 0:
                    next_player = WHITE if node.current_player == BLACK else BLACK

                    pass_child = Node(
                        (node.black_bb, node.white_bb, next_player),
                        node.args,
                        parent=node,
                        action_taken=-1,   # pass
                        prior=1 #100% pick rate since its the only option
                    )

                    node.children.append(pass_child)
                    node = pass_child


                # 2. Expansion using the policy network
                value = self.expand_node(node)

            # Backpropagate the value (value is from network or terminal evaluation)
            node.backpropagate(value)

        policy = np.zeros(64, dtype=np.float32)
        for child in root.children:
            if child.action_taken >= 0:  # exclude pass moves
                policy[child.action_taken] = child.visit_count
 
        total = policy.sum()
        if total > 0:
            policy /= total  # normalise to probability distribution
 
        move = self.choose_move(root, self.args['temperature'])
        return move, policy

    #MCTS functions but for BATCHES of games at once, increasing efficiency and speed
    def expand_batch(self, nodes: list) -> list:
        if not nodes:
            return []

        positions = [(n.black_bb, n.white_bb, n.current_player) for n in nodes]
        policies, values = self.model.inference_batch(positions, self.device)

        for i, node in enumerate(nodes):
            if not node.expandable_moves:
                node.expandable_moves = []
                continue
            self._attach_children(node, policies[i])

        return list(values)


    def search_batch(self, games: list, temperatures: list = None) -> list:
        # Returns [(move, policy), ...] for each game — same format as search()
        # temperatures: optional per-game temperature list (defaults to args["temperature"])
        roots = [
            Node((g.black_bb, g.white_bb, g.current_player), self.args)
            for g in games
        ]

        self.expand_batch(roots)
        for root in roots:
            if root.children:
                self.add_dirichlet_noise(root)

        #Main loop
        for _ in range(self.args['num_searches']):
            # 1. SELECTION
            leaves = []
            for root in roots:
                node = root
                while not node.is_terminal and node.is_expanded:
                    node = node.select_child()

                leaves.append(node)

            # 2. CHECK TERMINAL + MUST-PASS
            to_expand = []
            terminal_backprop = []

            for node in leaves:
                value, is_terminal = get_value_and_terminated(
                    node.black_bb, node.white_bb, node.current_player)
                if is_terminal:
                    terminal_backprop.append((node, float(value)))
                elif len(node.expandable_moves) == 0:
                    next_player = WHITE if node.current_player == BLACK else BLACK
                    pass_child = Node(
                        (node.black_bb, node.white_bb, next_player),
                        node.args, parent=node, action_taken=-1, prior=1
                    )
                    node.children.append(pass_child)
                    to_expand.append(pass_child)
                else:
                    to_expand.append(node)

            # 3. BATCH EXPAND
            if to_expand:
                exp_values = self.expand_batch(to_expand)
                for node, value in zip(to_expand, exp_values):
                    node.backpropagate(value)

            for node, value in terminal_backprop:
                node.backpropagate(value)

        #Makes sure the MCTS search is working properly, not short circuiting
        for i, root in enumerate(roots):
            assert root.visit_count == self.args["num_searches"], \
                f"Game {i}: expected {self.args['num_searches']} visits, got {root.visit_count}"

        results = []
        for i, root in enumerate(roots):
            policy = np.zeros(64, dtype=np.float32)
            for child in root.children:
                if child.action_taken >= 0:
                    policy[child.action_taken] = child.visit_count
            total = policy.sum()
            if total > 0:
                policy /= total
            temp = temperatures[i] if temperatures is not None else self.args['temperature']
            move = self.choose_move(root, temp)
            results.append((move, policy))

        return results