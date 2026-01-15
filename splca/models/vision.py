# ============================================================================
# splca/models/vision.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCAConv2d, SPLCALinear
from ..modulation import SPLCAModulator


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
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SPLCAConv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = SPLCAConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        # FIXED: Compute correct flattened size
        # input_size → conv1 → pool → conv2 → pool
        # 28 → 28 → 14 → 14 → 7
        size_after_pools = input_size // 4
        flattened_size = 64 * size_after_pools * size_after_pools
        
        self.fc1 = SPLCALinear(flattened_size, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.splca_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.modulator = SPLCAModulator(mode='validation')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def apply_splca_updates(self, val_loss: float, learning_rate: float = 1e-3):
        modulatory_scalar = self.modulator(val_loss)
        for layer in self.splca_layers:
            layer.update_eligibility_trace()
            layer.apply_splca_update(learning_rate, modulatory_scalar)
