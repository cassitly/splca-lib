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
        input_size: int = 28,  # FIXED: Added parameter with correct default
    ):
        super().__init__()
        self.conv1 = SPLCAConv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = SPLCAConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # FIXED: Compute correct flattened size
        # input_size → conv1 → pool → conv2 → pool
        # 28 → 28 → 14 → 14 → 7
        size_after_pools = input_size // 4
        flattened_size = 64 * size_after_pools * size_after_pools
        
        self.fc1 = SPLCALinear(flattened_size, hidden_dim)
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
        # FIXED: Predict current from previous
        for layer in self.splca_layers:
            if layer.current_output is not None and layer.prev_output is not None:
                error = layer.compute_local_error(layer.current_output)
                updates.append(layer.get_update_dict(error))
        return updates
