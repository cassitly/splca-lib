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
            
            # Process dimensions
            if error.dim() > 1:
                error_mean = error.mean(0)
            else:
                error_mean = error
            
            if presyn.dim() > 1:
                presyn_mean = presyn.mean(0)
            else:
                presyn_mean = presyn
            
            # CRITICAL FIX: Handle trace update based on parameter type
            trace = self.traces[param_id]
            
            if param.dim() == 2:  # Linear layer: (out_features, in_features)
                # Ensure presyn_mean matches in_features
                if presyn_mean.numel() >= param.size(1):
                    presyn_update = presyn_mean.flatten()[:param.size(1)]
                else:
                    presyn_update = torch.zeros(param.size(1), device=param.device)
                    presyn_update[:presyn_mean.numel()] = presyn_mean.flatten()
                
                # Broadcast to trace shape
                presyn_broadcast = presyn_update.view(1, -1).expand(param.size(0), -1)
                trace.trace = self.gamma * trace.trace + presyn_broadcast.detach()
                
            elif param.dim() == 4:  # Conv layer: (out_channels, in_channels, kH, kW)
                # For conv layers, just use a scalar update per weight
                # Simplified: average the input and broadcast
                presyn_scalar = presyn_mean.mean()
                trace.trace = self.gamma * trace.trace + presyn_scalar.detach()
            else:
                # Default: try to match shapes
                try:
                    presyn_reshaped = presyn_mean.view_as(trace.trace)
                    trace.trace = self.gamma * trace.trace + presyn_reshaped.detach()
                except:
                    # Fallback: use scalar
                    presyn_scalar = presyn_mean.mean()
                    trace.trace = self.gamma * trace.trace + presyn_scalar.detach()
            
            # Compute SPLCA update: Δw = -η·m·e·E
            if param.dim() == 2:  # Linear layer
                # Ensure error_mean matches out_features
                if error_mean.numel() >= param.size(0):
                    error_update = error_mean.flatten()[:param.size(0)]
                else:
                    error_update = torch.zeros(param.size(0), device=param.device)
                    error_update[:error_mean.numel()] = error_mean.flatten()
                
                # Outer product-like update
                error_expanded = error_update.view(-1, 1)
                delta_w = -self.lr * self.modulatory_scalar * (error_expanded * trace.trace)
                
            elif param.dim() == 4:  # Conv layer
                # Element-wise for conv
                if error_mean.numel() >= param.size(0):
                    error_update = error_mean.flatten()[:param.size(0)]
                else:
                    error_update = torch.zeros(param.size(0), device=param.device)
                    error_update[:error_mean.numel()] = error_mean.flatten()
                
                # Broadcast error to all kernel positions
                error_expanded = error_update.view(-1, 1, 1, 1).expand_as(param)
                delta_w = -self.lr * self.modulatory_scalar * (error_expanded * trace.trace)
            else:
                # Default element-wise
                delta_w = -self.lr * self.modulatory_scalar * (error_mean.view_as(param) * trace.trace)
            
            # Hebbian term
            if postsyn is not None:
                if postsyn.dim() > 1:
                    postsyn_mean = postsyn.mean(0)
                else:
                    postsyn_mean = postsyn
                
                if param.dim() == 2:
                    # Ensure dimensions match
                    if postsyn_mean.numel() >= param.size(0):
                        post_update = postsyn_mean.flatten()[:param.size(0)]
                    else:
                        post_update = torch.zeros(param.size(0), device=param.device)
                        post_update[:postsyn_mean.numel()] = postsyn_mean.flatten()
                    
                    if presyn_mean.numel() >= param.size(1):
                        pre_update = presyn_mean.flatten()[:param.size(1)]
                    else:
                        pre_update = torch.zeros(param.size(1), device=param.device)
                        pre_update[:presyn_mean.numel()] = presyn_mean.flatten()
                    
                    hebbian = -self.eta_heb * torch.outer(post_update, pre_update)
                    delta_w = delta_w + hebbian
            
            # Weight decay
            delta_w = delta_w - self.eta_stab * param.data
            
            # Apply update
            with torch.no_grad():
                param.data.add_(delta_w)
