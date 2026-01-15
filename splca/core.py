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
            
            # FIXED: Proper dimension handling
            if error.dim() > 1:
                error_mean = error.mean(0)
            else:
                error_mean = error
            
            if presyn.dim() > 1:
                presyn_mean = presyn.mean(0)
            else:
                presyn_mean = presyn
                
            # Update eligibility trace
            trace = self.traces[param_id]
            trace.trace = self.gamma * trace.trace + presyn_mean.view_as(trace.trace).detach()
            
            # Compute SPLCA update: Δw = -η·m·e·E
            if param.dim() == 2:  # Linear layer weight
                error_expanded = error_mean.view(-1, 1) if error_mean.dim() == 1 else error_mean
                trace_expanded = trace.trace.view(1, -1) if trace.trace.dim() == 1 else trace.trace
                
                # Match dimensions safely
                if error_expanded.size(0) != param.size(0):
                    error_expanded = error_expanded[:param.size(0)]
                if trace_expanded.size(1) != param.size(1):
                    trace_expanded = trace_expanded[:, :param.size(1)]
                    
                delta_w = -self.lr * self.modulatory_scalar * (error_expanded * trace_expanded.mean(0, keepdim=True))
            else:  # Conv weights
                delta_w = -self.lr * self.modulatory_scalar * trace.trace
            
            # Hebbian term
            if postsyn is not None:
                if postsyn.dim() > 1:
                    postsyn_mean = postsyn.mean(0)
                else:
                    postsyn_mean = postsyn
                    
                if param.dim() == 2:
                    post_expanded = postsyn_mean.view(-1, 1) if postsyn_mean.dim() == 1 else postsyn_mean
                    pre_expanded = presyn_mean.view(1, -1) if presyn_mean.dim() == 1 else presyn_mean
                    
                    if post_expanded.size(0) != param.size(0):
                        post_expanded = post_expanded[:param.size(0)]
                    if pre_expanded.size(1) != param.size(1):
                        pre_expanded = pre_expanded[:, :param.size(1)]
                        
                    hebbian = -self.eta_heb * (post_expanded * pre_expanded.mean(0, keepdim=True))
                    delta_w = delta_w + hebbian
            
            # Weight decay
            delta_w = delta_w - self.eta_stab * param.data
            
            # Apply update
            with torch.no_grad():
                param.data.add_(delta_w)
