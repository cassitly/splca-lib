# ============================================================================
# splca/models/audio.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear


class AudioClassifier(nn.Module):
    '''
    Audio classification using SPLCA linear layers.
    '''
    
    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.fc1 = SPLCALinear(input_dim, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, hidden_dim)
        self.fc3 = SPLCALinear(hidden_dim, num_classes)
        
        self.splca_layers = [self.fc1, self.fc2, self.fc3]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_splca_updates(self) -> list:
        updates = []
        # FIXED: Predict current from previous
        for layer in self.splca_layers:
            if layer.current_output is not None and layer.prev_output is not None:
                error = layer.compute_local_error(layer.current_output)
                updates.append(layer.get_update_dict(error))
        return updates
