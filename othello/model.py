import torch
import torch.nn as nn
import numpy as np
from numba import uint64
from board import BLACK, WHITE
from game import Game

#############################################################################
# Helpers

# x2 uint64 bitboards to 2 planes of uint8 arrays.
def bb_to_np(black_bb, white_bb):
    black_np = np.zeros(64, dtype=np.uint8)
    white_np = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        black_np[i] = (int(black_bb) >> i) & 1
        white_np[i] = (int(white_bb) >> i) & 1
    return black_np.reshape(8, 8), white_np.reshape(8, 8)

def bb_to_planes(black_bb, white_bb, turn: int):
    black_plane, white_plane = bb_to_np(black_bb, white_bb)
    
    if turn == BLACK:
        turn_plane = np.ones((8, 8), dtype=np.uint8)
        board = np.stack([black_plane, 
                          white_plane, 
                          turn_plane], axis=0)
    else:
        turn_plane = np.zeros((8, 8), dtype=np.uint8)
        board = np.stack([white_plane, 
                          black_plane, 
                          turn_plane], axis=0)
    
    return board

def bb_to_tensor(black_bb, white_bb, turn: int, device="mps"):
    board = bb_to_planes(black_bb, white_bb, turn)
    tensor = torch.tensor(board, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)

def board_to_planes(game: Game):
    return bb_to_planes(game.black_bb, game.white_bb, game.current_player)

def board_to_tensor(game: Game):
    return bb_to_tensor(game.black_bb, game.white_bb, game.current_player)
    

#############################################################################
# The architecture

# Entry block: Conv -> Batchnorm -> ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.block(X)

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters)
        )

    def forward(self, X):
        return nn.functional.relu(X + self.block(X), inplace=True)
    
class PolicyHead(nn.Module):
    def __init__(self, filters, board_size=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(self, X: torch.Tensor):
        X = self.conv(X)
        X = X.flatten(start_dim=1) # (batch, 2*8*8) => (batch, 128)
        return torch.softmax(self.fc(X), dim=1) # (batch, 64)
    
class ValueHead(nn.Module):
    def __init__(self, filters, board_size=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(board_size * board_size, board_size * board_size),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, X: torch.Tensor):
        X = self.conv(X)
        X = X.flatten(start_dim=1) # (batch, 64)
        return self.fc(X)


class ResNet(nn.Module):
    # AlphaZero style ResNet for Othello
    #
    ################################# Args #################################
    # n_res_blocks: Number of residual blocks (default 10)
    #               AlphaZero used 19-39 for chess/Go, but othello is simpler
    # filters: Feature depth throughout the trunk (default 128)
    def __init__(self, n_res_blocks=10, filters=128):
        super().__init__()
        
        self.entry = ConvBlock(in_channels=3, filters=filters)

        self.trunk = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(n_res_blocks)]
        )

        self.policy_head = PolicyHead(filters)
        self.value_head  = ValueHead(filters)

    def forward(self, X):
        X = self.entry(X) # (batch, filters, 8, 8)
        X = self.trunk(X) # (batch, filters, 8, 8)
        policy = self.policy_head(X)  # (batch, 64)
        value  = self.value_head(X)   # (batch, 1)
        return policy, value

    def inference(self, black_bb, white_bb, turn, device="mps"):
        self.eval()
        with torch.no_grad():
            tensor = bb_to_tensor(black_bb, white_bb, turn, device)
            policy, value = self.forward(tensor)
            policy_probs = policy.squeeze(0).numpy()
            value = value.item()
            return policy_probs, value # (,64), (1)

