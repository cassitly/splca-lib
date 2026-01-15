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
        
        self.prev_output = None
        self.current_input = None
        self.current_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_input = x.detach()
        output = self.linear(x)
        
        # Store previous before updating current
        self.prev_output = self.current_output
        self.current_output = output.detach()
        return output
    
    def predict_next(self) -> Optional[torch.Tensor]:
        '''Predict next activation using previous output'''
        if self.prev_output is None:
            return None
        return self.predictor(self.prev_output)
    
    def compute_local_error(self, target_activation: torch.Tensor) -> torch.Tensor:
        '''Compute local prediction error: e = y(t) - ŷ(t) predicted from y(t-1)'''
        predicted = self.predict_next()
        if predicted is None:
            return torch.zeros_like(target_activation)

        # Ensure sizes match
        if predicted.size() != target_activation.size():
            min_size = [min(predicted.size(i), target_activation.size(i)) for i in range(len(predicted.size()))]
            predicted = predicted[tuple(slice(0, s) for s in min_size)]
            target_activation = target_activation[tuple(slice(0, s) for s in min_size)]

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
    For vision tasks - uses spatial predictors.
    '''
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        predictor_hidden: int = 16,  # Smaller for conv
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        # Predictor uses 1x1 conv - predicts same spatial size
        self.predictor = nn.Sequential(
            nn.Conv2d(out_channels, predictor_hidden, 1),
            nn.ReLU(),
            nn.Conv2d(predictor_hidden, out_channels, 1)
        )
        
        self.prev_output = None
        self.current_input = None
        self.current_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_input = x.detach()
        output = self.conv(x)
        
        # Store previous before updating
        self.prev_output = self.current_output
        self.current_output = output.detach()
        return output
    
    def predict_next(self) -> Optional[torch.Tensor]:
        if self.prev_output is None:
            return None
        return self.predictor(self.prev_output)
    
    def compute_local_error(self, target_activation: torch.Tensor) -> torch.Tensor:
        '''Compute spatial prediction error'''
        predicted = self.predict_next()
        if predicted is None:
            return torch.zeros_like(target_activation)

        # Ensure sizes match
        if predicted.size() != target_activation.size():
            min_size = [min(predicted.size(i), target_activation.size(i)) for i in range(len(predicted.size()))]
            predicted = predicted[tuple(slice(0, s) for s in min_size)]
            target_activation = target_activation[tuple(slice(0, s) for s in min_size)]

        return target_activation.detach() - predicted
    
    def get_update_dict(self, error: torch.Tensor) -> dict:
        '''
        Package data for SPLCA optimizer.
        For conv layers, we aggregate spatially.
        '''
        # error shape: (batch, out_channels, H, W)
        # Aggregate over batch and spatial dimensions
        error_aggregated = error.mean(dim=[0, 2, 3])  # (out_channels,)
        
        # Input aggregated
        input_aggregated = self.current_input.mean(dim=[0, 2, 3])  # (in_channels,)
        
        # Output aggregated
        output_aggregated = self.current_output.mean(dim=[0, 2, 3])  # (out_channels,)
        
        return {
            'param': self.conv.weight,
            'error': error_aggregated,
            'presyn': input_aggregated,
            'postsyn': output_aggregated,
        }
