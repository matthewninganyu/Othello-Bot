### ResNet Input ###

Two uint64 ints
────────────────────────────────────────────
black_bb = uint64(...)   e.g. bits 28,35 set
white_bb = uint64(...)   e.g. bits 27,36 set

        ↓  int() + bit extraction loop

Two flat np.uint8 arrays, shape (64,)
────────────────────────────────────────────
black_plane = [0,0,...,1,...,1,...,0]   64 values, 0 or 1
white_plane = [0,0,...,1,...,1,...,0]   64 values, 0 or 1

        ↓  stack + reshape to (2, 64) → (2, 8, 8)

Two spatial planes, shape (2, 8, 8)
────────────────────────────────────────────
black_plane:               white_plane:
. . . . . . . .            . . . . . . . .
. . . . . . . .            . . . . . . . .
. . . . . . . .            . . . . . . . .
. . . . ● . . .            . . . ○ . . . .
. . . ● . . . .            . . . . ○ . . .
. . . . . . . .            . . . . . . . .
. . . . . . . .            . . . . . . . .
. . . . . . . .            . . . . . . . .

        ↓  add 3rd turn plane (all 1s if black to move, all 0s if white)

Three spatial planes, shape (3, 8, 8)
────────────────────────────────────────────
plane 0: black pieces
plane 1: white pieces
plane 2: turn indicator (uniform 1 or 0)

        ↓  torch.tensor(..., dtype=torch.float32)
        ↓  unsqueeze(0) to add batch dim

Final CNN input, shape (1, 3, 8, 8)
────────────────────────────────────────────
(batch, channels, height, width)  ← what ResNet expects