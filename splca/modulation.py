# ============================================================================
# splca/modulation.py
# ============================================================================
import torch
import torch.nn as nn
from collections import deque


class ValidationModulator:
    '''
    Compute modulatory scalar from validation loss improvement.
    m(t) = sigmoid(α·(-ΔL_val))
    '''
    
    def __init__(self, alpha: float = 1.0, window: int = 10):
        self.alpha = alpha
        self.loss_history = deque(maxlen=window)
        
    def __call__(self, val_loss: float) -> float:
        self.loss_history.append(val_loss)
        
        if len(self.loss_history) < 2:
            return 0.5
        
        # Compute improvement
        delta_loss = self.loss_history[-2] - self.loss_history[-1]
        m = torch.sigmoid(torch.tensor(self.alpha * delta_loss)).item()
        return m


class RewardModulator:
    '''
    Compute modulatory scalar from reward signal (RL setting).
    '''
    
    def __init__(self, alpha: float = 1.0, baseline_momentum: float = 0.9):
        self.alpha = alpha
        self.baseline = 0.0
        self.momentum = baseline_momentum
        
    def __call__(self, reward: float) -> float:
        # Update baseline
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward
        
        # Advantage
        advantage = reward - self.baseline
        m = torch.sigmoid(torch.tensor(self.alpha * advantage)).item()
        return m


class LearnedModulator(nn.Module):
    '''
    Learned modulator network (small MLP).
    Takes recent statistics and outputs m(t).
    '''
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, stats: torch.Tensor) -> float:
        '''stats: tensor of recent metrics (losses, accuracies, etc.)'''
        return self.net(stats).item()