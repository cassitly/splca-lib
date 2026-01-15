# ============================================================================
# splca/core.py
# ============================================================================
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable


class EligibilityTrace:
    '''Manages eligibility traces for temporal credit assignment.'''
    
    def __init__(self, shape: tuple, gamma: float = 0.95, device: str = 'cpu'):
        self.gamma = gamma
        self.trace = torch.zeros(shape, device=device)
        
    def update(self, presynaptic: torch.Tensor) -> torch.Tensor:
        '''Update trace: E(t+1) = γ*E(t) + x(t)'''
        self.trace = self.gamma * self.trace + presynaptic.detach()
        return self.trace
    
    def reset(self):
        '''Reset trace to zeros'''
        self.trace.zero_()


class SPLCAOptimizer:
    '''
    Self-Predictive Local Credit Assignment optimizer.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (η)
        gamma: Eligibility trace decay factor
        eta_pred: Learning rate for predictors
        eta_heb: Hebbian term coefficient
        eta_stab: Weight stabilization (decay) coefficient
    '''
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 0.95,
        eta_pred: float = 1e-3,
        eta_heb: float = 1e-4,
        eta_stab: float = 1e-5,
    ):
        self.param_groups = [{'params': list(params)}]
        self.lr = lr
        self.gamma = gamma
        self.eta_pred = eta_pred
        self.eta_heb = eta_heb
        self.eta_stab = eta_stab
        
        # Storage for traces and state
        self.traces: Dict[int, EligibilityTrace] = {}
        self.prev_activations: Dict[int, torch.Tensor] = {}
        self.modulatory_scalar = 0.5
        
    def zero_grad(self):
        '''Zero out gradients (compatibility with PyTorch)'''
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
    
    def set_modulation(self, m: float):
        '''Set global modulatory scalar m(t)'''
        self.modulatory_scalar = max(0.0, min(1.0, m))
    
    def step(self, layer_updates: List[Dict]):
        '''
        Apply SPLCA updates.
        
        Args:
            layer_updates: List of dicts with keys:
                - 'param': parameter tensor
                - 'error': local prediction error
                - 'presyn': presynaptic activations
                - 'postsyn': postsynaptic activations
        '''
        for update_dict in layer_updates:
            param = update_dict['param']
            error = update_dict['error']
            presyn = update_dict['presyn']
            postsyn = update_dict.get('postsyn', None)
            
            param_id = id(param)
            
            # Initialize trace if needed
            if param_id not in self.traces:
                self.traces[param_id] = EligibilityTrace(
                    param.shape, self.gamma, param.device
                )
            
            trace = self.traces[param_id]
            
            # Aggregate batch dimension for all inputs
            if error.dim() > 1:
                error = error.mean(0)
            if presyn.dim() > 1:
                presyn = presyn.mean(0)
            if postsyn is not None and postsyn.dim() > 1:
                postsyn = postsyn.mean(0)
            
            # Flatten all tensors
            error = error.flatten()
            presyn = presyn.flatten()
            if postsyn is not None:
                postsyn = postsyn.flatten()
            
            # Handle different parameter types
            if param.dim() == 2:  # Linear layer: (out_features, in_features)
                out_features, in_features = param.shape
                
                # Ensure error matches out_features
                if error.numel() < out_features:
                    error_padded = torch.zeros(out_features, device=param.device)
                    error_padded[:error.numel()] = error
                    error = error_padded
                else:
                    error = error[:out_features]
                
                # Ensure presyn matches in_features
                if presyn.numel() < in_features:
                    presyn_padded = torch.zeros(in_features, device=param.device)
                    presyn_padded[:presyn.numel()] = presyn
                    presyn = presyn_padded
                else:
                    presyn = presyn[:in_features]
                
                # Update trace: broadcast presyn to match param shape
                presyn_broadcast = presyn.view(1, -1).expand(out_features, -1)
                trace.trace = self.gamma * trace.trace + presyn_broadcast.detach()
                
                # Compute SPLCA update: Δw = -η·m·e·E
                error_broadcast = error.view(-1, 1).expand(out_features, in_features)
                delta_w = -self.lr * self.modulatory_scalar * (error_broadcast * trace.trace)
                
                # Hebbian term: -η_heb·(y⊗x)
                if postsyn is not None:
                    # Ensure postsyn matches out_features
                    if postsyn.numel() < out_features:
                        postsyn_padded = torch.zeros(out_features, device=param.device)
                        postsyn_padded[:postsyn.numel()] = postsyn
                        postsyn = postsyn_padded
                    else:
                        postsyn = postsyn[:out_features]
                    
                    hebbian = -self.eta_heb * torch.outer(postsyn, presyn)
                    delta_w = delta_w + hebbian
                
            elif param.dim() == 4:  # Conv layer: (out_channels, in_channels, kH, kW)
                out_ch, in_ch, kH, kW = param.shape
                
                # For conv, use scalar updates (simplified)
                error_scalar = error.mean() if error.numel() > 0 else torch.tensor(0.0, device=param.device)
                presyn_scalar = presyn.mean() if presyn.numel() > 0 else torch.tensor(0.0, device=param.device)
                
                # Update trace with scalar
                trace.trace = self.gamma * trace.trace + presyn_scalar.detach()
                
                # Scalar update broadcasted
                delta_w = -self.lr * self.modulatory_scalar * error_scalar * trace.trace
                
                # Simplified Hebbian for conv
                if postsyn is not None:
                    postsyn_scalar = postsyn.mean() if postsyn.numel() > 0 else torch.tensor(0.0, device=param.device)
                    hebbian_scalar = -self.eta_heb * postsyn_scalar * presyn_scalar
                    delta_w = delta_w + hebbian_scalar
            
            else:
                # For other param types, use element-wise update
                error_scalar = error.mean()
                trace.trace = self.gamma * trace.trace + presyn.mean().detach()
                delta_w = -self.lr * self.modulatory_scalar * error_scalar * trace.trace
            
            # Weight stabilization (decay)
            delta_w = delta_w - self.eta_stab * param.data
            
            # Apply update
            with torch.no_grad():
                param.data.add_(delta_w)
