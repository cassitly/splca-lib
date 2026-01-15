# ============================================================================
# splca/layers.py
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SPLCALinear(nn.Module):
    '''
    Linear layer with SPLCA support.
    Maintains activations and provides hooks for local updates.
    '''
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        predictor_type: str = 'linear',
        predictor_hidden: int = 64,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Create predictor network
        if predictor_type == 'linear':
            self.predictor = nn.Linear(out_features, out_features, bias=False)
        elif predictor_type == 'mlp':
            self.predictor = nn.Sequential(
                nn.Linear(out_features, predictor_hidden),
                nn.ReLU(),
                nn.Linear(predictor_hidden, out_features)
            )
        
        self.prev_output = None  # FIXED: Added this line
        self.current_input = None
        self.current_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_input = x.detach()
        output = self.linear(x)
        
        # FIXED: Store previous before updating current
        self.prev_output = self.current_output
        self.current_output = output.detach()
        return output
    
    def predict_next(self) -> Optional[torch.Tensor]:
        '''Predict next activation using previous output'''
        if self.prev_output is None:  # FIXED: Use prev_output
            return None
        return self.predictor(self.prev_output)
    
    def compute_local_error(self, target_activation: torch.Tensor) -> torch.Tensor:
        '''Compute local prediction error: e = y(t) - ŷ(t) predicted from y(t-1)'''
        predicted = self.predict_next()
        if predicted is None:
            return torch.zeros_like(target_activation)
        return target_activation.detach() - predicted
    
    def get_update_dict(self, error: torch.Tensor) -> dict:
        '''Package data for SPLCA optimizer'''
        return {
            'param': self.linear.weight,
            'error': error,
            'presyn': self.current_input,
            'postsyn': self.current_output,
        }


class SPLCAConv2d(nn.Module):
    '''
    Convolutional layer with SPLCA support.
    For vision tasks.
    '''
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        predictor_hidden: int = 64,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        # Predictor uses 1x1 conv for efficiency
        self.predictor = nn.Sequential(
            nn.Conv2d(out_channels, predictor_hidden, 1),
            nn.ReLU(),
            nn.Conv2d(predictor_hidden, out_channels, 1)
        )
        
        self.prev_output = None  # FIXED: Added this line
        self.current_input = None
        self.current_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_input = x.detach()
        output = self.conv(x)
        
        # FIXED: Store previous before updating current
        self.prev_output = self.current_output
        self.current_output = output.detach()
        return output
    
    def predict_next(self) -> Optional[torch.Tensor]:
        if self.prev_output is None:  # FIXED: Use prev_output
            return None
        return self.predictor(self.prev_output)
    
    def compute_local_error(self, target_activation: torch.Tensor) -> torch.Tensor:
        predicted = self.predict_next()
        if predicted is None:
            return torch.zeros_like(target_activation)
        return target_activation.detach() - predicted
    
    def get_update_dict(self, error: torch.Tensor) -> dict:
        # Reshape for outer product computation
        error_flat = error.flatten(1).mean(0)
        input_flat = self.current_input.flatten(1).mean(0)
        return {
            'param': self.conv.weight,
            'error': error_flat[:self.conv.out_channels],
            'presyn': input_flat[:self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]],
            'postsyn': self.current_output.flatten(1).mean(0)[:self.conv.out_channels],
        }
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        predictor_hidden: int = 64,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        # Predictor uses 1x1 conv for efficiency
        self.predictor = nn.Sequential(
            nn.Conv2d(out_channels, predictor_hidden, 1),
            nn.ReLU(),
            nn.Conv2d(predictor_hidden, out_channels, 1)
        )
        
        self.current_input = None
        self.current_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_input = x.detach()
        output = self.conv(x)
        self.current_output = output.detach()
        return output
    
    def predict_next(self) -> Optional[torch.Tensor]:
        if self.current_output is None:
            return None
        return self.predictor(self.current_output)
    
    def compute_local_error(self, next_activation: torch.Tensor) -> torch.Tensor:
        predicted = self.predict_next()
        if predicted is None:
            return torch.zeros_like(next_activation)
        return next_activation.detach() - predicted
    
    def get_update_dict(self, error: torch.Tensor) -> dict:
        # Reshape for outer product computation
        error_flat = error.flatten(1).mean(0)
        input_flat = self.current_input.flatten(1).mean(0)
        return {
            'param': self.conv.weight,
            'error': error_flat[:self.conv.out_channels],
            'presyn': input_flat[:self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]],
            'postsyn': self.current_output.flatten(1).mean(0)[:self.conv.out_channels],
        }

