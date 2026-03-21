# `python -m othello.train` — Full Pipeline Reference

Top-down walkthrough of every function called when the training process runs.
Line numbers reference the source files as of the initial implementation.

---

## 1. Entry Point (`train.py`, line 322)

```python
if __name__ == "__main__":
```

Steps executed at startup:

1. **Device selection** — hardcoded `device = "cuda"`; prints GPU name via `torch.cuda.get_device_name(0)`
2. **Model instantiation** — `ResNet(n_res_blocks=10, filters=128).to(device)`
3. **Optional checkpoint resume** — if `RESUME_CHECKPOINT` path exists, calls `torch.load(path, map_location=device)` and loads into model via `model.load_state_dict(...)`
4. **Optimizer creation** — `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`
5. **Hand off** — calls `training_loop(model, optimizer, CONFIG, device)`

**`CONFIG` snapshot (defaults):**

| Key | Value | Notes |
|-----|-------|-------|
| `n_res_blocks` | 10 | ResNet depth |
| `filters` | 128 | Feature channels |
| `n_iterations` | 1000 | Total training iterations |
| `n_self_play_games` | 100 | Escalates — see §2 |
| `num_searches` | 150 | MCTS simulations per move — escalates |
| `temperature` | 1.0 | Move sampling temp |
| `temperature_drop` | 20 | Move count after which temp → 0 |
| `buffer_max_games` | 2000 | Max games in replay buffer |
| `n_epochs` | 100 | Gradient steps per iteration |
| `batch_size` | 256 | Samples per gradient step |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-4 | Adam L2 regularisation |
| `n_eval_games` | 25 | Per-colour evaluation games (50 total) |
| `win_rate_threshold` | 0.52 | Threshold to promote new model |
| `dirichlet_alpha` | 0.3 | Dirichlet noise concentration |
| `dirichlet_epsilon` | 0.25 | Noise mix fraction at root |
| `exploration_constant` | 1.5 | PUCT exploration weight |
| `checkpoint_every` | 5 | Save checkpoint every N iterations |

---

## 2. `training_loop()` (`train.py`, line 261)

### Setup (runs once)

- Creates `ReplayBuffer(max_pos = 2000 * 60 * 8 = 960_000)` — fixed-window deque
- Creates timestamped run directory: `checkpoints/YYYYMMDD_HHMMSS/`
- Opens `loss_log.csv` in that directory (columns: `iteration, policy_loss, value_loss, total_loss`)
- Clones model into `best_model`: `ResNet(...).load_state_dict(model.state_dict())` — frozen reference

### Per-iteration loop (`iteration = 1 → n_iterations`)

**Hyperparameter escalation** (overrides `CONFIG` in-place):

| Iteration range | `num_searches` | `n_self_play_games` |
|-----------------|---------------|---------------------|
| 1–25            | 150           | 100                 |
| 26–150          | 250           | 150                 |
| 151+            | 400           | 200                 |

**Phase sequence per iteration:**

1. `run_self_play(buffer, model, args, device)` → populates buffer
2. `run_training(model, optimizer, buffer, args, device)` → gradient updates
3. If `iteration > 20` (post-warmup): `evaluate_models(model, best_model, args, device)`
   - `win_rate >= 0.52` → promote: `best_model.load_state_dict(model.state_dict())`
   - else → revert: `model.load_state_dict(best_model.state_dict())`
4. If `iteration % 5 == 0` → `torch.save(model.state_dict(), run_dir/model_iterN.pt)`

---

## 3. `run_self_play()` (`train.py`, line 106)

Runs all `n_self_play_games` simultaneously, sharing one batched NN forward pass per MCTS step.

### Setup

```python
games       = [Game() for _ in range(n)]   # fresh Othello boards
histories   = [[] for _ in range(n)]        # (board_planes, policy, player, move_count)
move_counts = [0] * n
mcts        = MCTS(games[0], model, args, device)  # one shared MCTS instance
model.eval()
```

### Game loop (`while any game is active`)

Each iteration:

1. Finds all active game indices (not `game_over`)
2. Computes per-game temperature:
   - `T = args["temperature"]` if `move_count < temperature_drop (20)`
   - `T = 0` (greedy) otherwise
3. Calls `mcts.search_batch([active_games], temps)` → `[(move, policy), ...]`
4. For each active game:
   - Records `(board_to_planes(game), policy, current_player, move_count)` in history
   - Calls `game.make_move(move)` → updates bitboards; `Game` handles pass turns internally
   - Increments `move_counts[i]`

### Post-game value labeling

For each finished game:

```python
winner = games[i].winner  # BLACK, WHITE, or None (draw)
val = 0 if winner is None else (1 if winner == player else -1)
```

Only positions after move 3 (`turn > 3`) are kept (avoids near-identical opening states).

### 8× symmetry augmentation

For each kept position, calls `get_rotations(board_planes, policy)`:
- 4 rotations × 2 mirrors (horizontal flip) = 8 augmented samples
- Both board tensor (axes 1,2) and policy (reshaped to 8×8) are transformed together

Results pushed to buffer via `buffer.add_game(samples)`.

---

## 4. `MCTS.search_batch()` (`BetaFish.py`, line 383)

Core MCTS routine. Returns `[(move, policy), ...]` for each game in the input list.

### Initialization

```python
roots = [Node((g.black_bb, g.white_bb, g.current_player), args) for g in games]
```

- Each `Node` stores raw bitboards (`np.uint64`), `current_player`, and computes `expandable_moves` via `get_moves()`
- Calls `expand_batch(roots)` — **first NN forward pass** to prime root priors
- Calls `add_dirichlet_noise(root)` on each root with children:
  - Skipped if `dirichlet_epsilon == 0` (e.g., during evaluation)
  - Samples `Dirichlet(alpha=0.3)` over child count, mixes: `prior = 0.75 * prior + 0.25 * noise`

### Search loop (`num_searches` iterations)

#### Step 1 — Selection

For each root, walk down the tree using PUCT until reaching a leaf (not expanded or terminal):

```python
while not node.is_terminal and node.is_expanded:
    node = node.select_child()
```

`select_child()` picks the child maximising:

```
PUCT = -(child.value_sum / child.visit_count)      # exploitation (negated: child is opponent)
     + C * child.prior * sqrt(N_parent) / (1 + N_child)  # exploration (C = 1.5)
```

`is_expanded` is `True` when `len(children) > 0 and len(expandable_moves) == 0`.

#### Step 2 — Terminal / pass check

For each leaf node, calls `get_value_and_terminated(black_bb, white_bb, current_player)`:

| Leaf state | Action |
|-----------|--------|
| Terminal (both players have no moves) | Queue `(node, value)` for immediate backprop |
| Must-pass (current player has no moves, non-terminal) | Create `Node(action_taken=-1, prior=1)`, queue for expand |
| Normal unexpanded leaf | Queue for expand |

#### Step 3 — Batch expand

```python
exp_values = self.expand_batch(to_expand)
```

One NN forward pass for all queued leaves (see §5).

#### Step 4 — Backpropagation

`node.backpropagate(value)` walks up to the root:

```python
while node is not None:
    node.value_sum += value
    node.visit_count += 1
    node = node.parent
    value = -value          # flip perspective at each level
```

Terminal backprops happen in the same iteration after batch expand.

### Post-search assertions and output

```python
assert root.visit_count == num_searches  # sanity check: no short-circuits
```

For each root:
- Builds `policy[64]` from child visit counts (pass moves excluded: `action_taken >= 0`)
- Normalises to sum-to-1
- Calls `choose_move(root, temperature)`:
  - `T = 0` → greedy: `argmax(visit_counts)`
  - `T > 0` → sample ∝ `visit_count^(1/T)`

---

## 5. `MCTS.expand_batch()` (`BetaFish.py`, line 367)

Single GPU forward pass for a batch of nodes.

```python
positions = [(n.black_bb, n.white_bb, n.current_player) for n in nodes]
policies, values = self.model.inference_batch(positions, self.device)
```

**`model.inference_batch()`** (`model.py`, line 149):
1. Stacks positions: `np.stack([bb_to_planes(b, w, t) for b, w, t in positions])` → `(N, 3, 8, 8)`
2. Converts to `torch.float32` tensor, moves to device
3. Calls `ResNet.forward(states)` under `torch.no_grad()`
4. Returns `policies.cpu().numpy()` shape `(N, 64)` and `values.squeeze(-1).cpu().numpy()` shape `(N,)`

**`bb_to_planes(black_bb, white_bb, turn)`** (`model.py`, line 20):
- Plane 0 = current player's pieces (black if BLACK's turn, white if WHITE's)
- Plane 1 = opponent's pieces
- Plane 2 = turn indicator (all 1.0 if BLACK, all 0.0 if WHITE)

**`ResNet.forward()`** (`model.py`, line 133):
```
Input (N, 3, 8, 8)
  → ConvBlock: Conv2d(3→128, k=3, pad=1) → BN → ReLU        → (N, 128, 8, 8)
  → 10× ResidualBlock: [Conv→BN→ReLU→Conv→BN] + residual skip → (N, 128, 8, 8)
  → PolicyHead:
      Conv2d(128→2, k=1) → BN → ReLU → flatten → Linear(128→64) → Softmax → (N, 64)
  → ValueHead:
      Conv2d(128→1, k=1) → BN → ReLU → flatten → Linear(64→64) → ReLU → Linear(64→1) → Tanh → (N, 1)
```

**`_attach_children(node, policy)`** (`BetaFish.py`, line 233):
- Extracts prior probs for legal moves only; renormalises (uniform fallback if all-zero)
- For each legal move `m`: calls `apply_move()` (Numba JIT bitboard update), determines `next_player` by checking if opponent has any moves after
- Creates a child `Node` with the computed prior and appends to `node.children`
- Clears `node.expandable_moves = []`

> **Known bug (TODO a):** When a node is a pass node (`expandable_moves` is already empty on entry), `expand_batch` skips `_attach_children`. The pass node never gets children, so subsequent selection loops may enter it repeatedly → potential infinite loop.

---

## 6. `run_training()` (`train.py`, line 158)

100 gradient steps using random samples from the replay buffer.

```python
model.train()
value_criterion = nn.MSELoss()
```

Per step:

1. `buffer.get_batch(256)` — `random.sample(buffer, min(256, len(buffer)))` → tensors on device
2. `pred_policy, pred_value = model(boards)` — forward pass
3. **Policy loss** (manual cross-entropy, since head outputs softmax not logits):
   ```python
   p_loss = -(target_policy * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
   ```
4. **Value loss**: `MSELoss(pred_value.squeeze(-1), target_value)`
5. `total_loss = p_loss + v_loss` (equal weighting)
6. `optimizer.zero_grad()` → `loss.backward()` → `clip_grad_norm_(max_norm=1.0)` → `optimizer.step()`

Returns average `(p_loss, v_loss)` over 100 steps, written to `loss_log.csv`.

---

## 7. `evaluate_models()` (`train.py`, line 199)

Runs `2 * n_eval_games = 50` games simultaneously. `new_model` plays each colour in half.

### Setup

```python
eval_args = {**args, "temperature": 0, "dirichlet_epsilon": 0}  # greedy, no noise
games        = [Game() for _ in range(50)]
new_plays_as = [BLACK] * 25 + [WHITE] * 25
new_mcts  = MCTS(games[0], new_model,  eval_args, device)
best_mcts = MCTS(games[0], best_model, eval_args, device)
```

### Game loop

Each iteration splits active games:

- **Opening phase** (`move_count < eval_opening_ply=4`): random legal move → position diversity
- **MCTS phase** (`move_count >= 4`):
  - `new_turn` = games where `current_player == new_plays_as[i]`
  - `best_turn` = games where `current_player != new_plays_as[i]`
  - Each group gets its own `search_batch` call (separate NN instances)

### Scoring

```python
wins += 1    if winner == new_plays_as[i] else
        0.5  if winner is None             else
        0
```

Returns `wins / 50`. Promotion threshold: `>= 0.52`.

---

## Data Flow Summary

```
__main__ (train.py:322)
  └─ training_loop() (train.py:261)
       ├─ [iter 1..N] run_self_play() (train.py:106)
       │    └─ MCTS.search_batch() (BetaFish.py:383)      ← batched call per move step
       │         ├─ expand_batch() [root init] (BetaFish.py:367)
       │         │    └─ model.inference_batch() (model.py:149)   ← GPU forward pass
       │         │         └─ ResNet.forward() (model.py:133)
       │         │              ├─ ConvBlock  (3 → 128 filters)
       │         │              ├─ 10× ResidualBlock (128 → 128)
       │         │              ├─ PolicyHead → softmax(64)
       │         │              └─ ValueHead  → tanh(1)
       │         └─ [num_searches iters]
       │              ├─ select_child() × N  (PUCT walk)
       │              ├─ get_value_and_terminated() (board.py — Numba JIT)
       │              ├─ expand_batch() [leaves] (BetaFish.py:367)
       │              │    └─ model.inference_batch()  ← GPU forward pass
       │              │         └─ _attach_children()   ← apply_move() Numba JIT
       │              └─ backpropagate() (sign flip per level)
       │
       ├─ [iter 1..N] run_training() (train.py:158)
       │    └─ [100 epochs]
       │         ├─ buffer.get_batch(256)
       │         ├─ model.forward()  ← GPU forward pass
       │         └─ loss.backward() + clip_grad_norm_ + optimizer.step()
       │
       └─ [iter >20]  evaluate_models() (train.py:199)
            ├─ [opening_ply=4 moves] random.choice(legal_moves)
            └─ MCTS.search_batch() for new_model and best_model (interleaved by turn)
```

---

## Critical Files

| File | Role |
|------|------|
| `othello/train.py` | Entry point; `training_loop`, `run_self_play`, `run_training`, `evaluate_models`, `ReplayBuffer`, `get_rotations` |
| `othello/BetaFish.py` | `Node` (PUCT, backprop) + `MCTS` (`search_batch`, `expand_batch`, `_attach_children`) |
| `othello/model.py` | `ResNet`, `inference_batch`, `bb_to_planes`, `ConvBlock`, `ResidualBlock`, `PolicyHead`, `ValueHead` |
| `othello/board.py` | Numba JIT `apply_move`, `move_gen`, `get_moves`, `get_value_and_terminated`, `popcount` |
| `othello/game.py` | `Game.make_move()`, `legal_moves`, `game_over`, `winner` |
