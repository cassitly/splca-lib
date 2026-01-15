# ============================================================================
# splca/models/audio.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear
from ..modulation import SPLCAModulator


class AudioClassifier(nn.Module):
    '''
    Audio classification using SPLCA linear layers.
    '''
    
    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.fc1 = SPLCALinear(input_dim, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, hidden_dim)
        self.fc3 = SPLCALinear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.splca_layers = [self.fc1, self.fc2, self.fc3]
        self.modulator = SPLCAModulator(mode='validation')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def apply_splca_updates(self, val_loss: float, learning_rate: float = 1e-3):
        modulatory_scalar = self.modulator(val_loss)
        for layer in self.splca_layers:
            layer.update_eligibility_trace()
            layer.apply_splca_update(learning_rate, modulatory_scalar)
