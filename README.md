# Othello Engine

## Project Dependencies (version python@3.12.11)

| Package | Install | Purpose |
|---|---|---|
| **PyTorch** | `pip install torch torchvision` | Neural network and such |
| **NumPy** | `pip install numpy` | Fast matrix / array ops |
| **Modal** | `pip install modal` | Rent cloud GPUs for self-play & training jobs |
| **Numba** | `pip install numba` | JIT-compile hot game logic loops for speed |
| **Weights & Biases** | `pip install wandb` | Track and visualize training runs |
| **pytest** | `pip install pytest` | Unit test game logic (move gen, flip rules, etc.) |
| **black** | `pip install black` | Auto-format code |
| **isort** | `pip install isort` | Sort and organize imports |

### To install run
```bash
pip install torch torchvision torchaudio numpy modal numba wandb cloudpickle pytest black isort
```
### or (after cloning):
```bash
pip install -r requirements.txt
```

> **GPU note:** For local CUDA, get the correct PyTorch wheel at [pytorch.org](https://pytorch.org/get-started/locally/). On Modal, the GPU is specified via decorator — no local CUDA needed.