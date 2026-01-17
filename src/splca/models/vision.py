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

    def get_splca_updates(self):
        '''
        Collect SPLCA updates from all layers.
        Returns list of dicts with update information.
        '''
        updates = []
        
        # Convolutional layers
        for layer in self.splca_conv_layers:
            if layer.current_output is not None:
                local_error = layer.compute_local_error(layer.current_output)
                update_dict = layer.get_update_dict(local_error)
                updates.append(update_dict)
        
        # Linear layers
        for layer in self.splca_linear_layers:
            local_error = layer.compute_local_error()
            if local_error is not None and layer.current_input is not None:
                error_agg = local_error.mean(dim=0)
                input_agg = layer.current_input.mean(dim=0)
                output_agg = layer.current_output.mean(dim=0) if layer.current_output is not None else None
                
                updates.append({
                    'param': layer.linear.weight,
                    'error': error_agg,
                    'presyn': input_agg,
                    'postsyn': output_agg,
                })
        
        return updates
