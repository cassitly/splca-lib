# ============================================================================
# splca/models/vision.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCAConv2d, SPLCALinear


class VisionClassifier(nn.Module):
    '''
    CNN for image classification using SPLCA layers.
    Conv → Conv → Pool → Linear → Linear → Output
    '''
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = SPLCAConv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = SPLCAConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute flattened size (depends on input resolution)
        # Assuming 32x32 input → after 2 pools: 8x8
        self.fc1 = SPLCALinear(64 * 8 * 8, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, num_classes)
        
        self.splca_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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