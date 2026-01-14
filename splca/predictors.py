# ============================================================================
# splca/predictors.py
# ============================================================================
import torch
import torch.nn as nn


class LinearPredictor(nn.Module):
    '''Simple linear predictor: ŷ = Wx'''
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPPredictor(nn.Module):
    '''MLP predictor with hidden layer'''
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPredictor(nn.Module):
    '''Predictor with temporal convolution for sequences'''
    
    def __init__(self, dim: int, history: int = 3):
        super().__init__()
        self.history = history
        self.conv = nn.Conv1d(dim, dim, kernel_size=history, padding=history-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq)
        out = self.conv(x)
        return out[:, :, :x.size(2)].transpose(1, 2)