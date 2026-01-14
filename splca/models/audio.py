# ============================================================================
# splca/models/audio.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear


class AudioClassifier(nn.Module):
    '''
    Audio classification using 1D convolutions + SPLCA linear layers.
    For waveform or spectrogram inputs.
    '''
    
    def __init__(
        self,
        input_dim: int = 128,  # e.g., mel bins
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Simple architecture: input → linear layers
        # (For real use, add Conv1d layers)
        self.fc1 = SPLCALinear(input_dim, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, hidden_dim)
        self.fc3 = SPLCALinear(hidden_dim, num_classes)
        
        self.splca_layers = [self.fc1, self.fc2, self.fc3]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) or (batch, features)
        if x.dim() == 3:
            x = x.mean(dim=1)  # temporal pooling
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_splca_updates(self) -> list:
        updates = []
        for i, layer in enumerate(self.splca_layers[:-1]):
            next_layer = self.splca_layers[i + 1]
            next_output = next_layer.current_output
            if next_output is not None:
                error = layer.compute_local_error(next_output)
                updates.append(layer.get_update_dict(error))
        return updates