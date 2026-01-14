# ============================================================================
# splca/models/text.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear


class TextClassifier(nn.Module):
    '''
    Simple text classification model using SPLCA layers.
    Embeddings → SPLCA Linear → SPLCA Linear → Output
    '''
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        max_length: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = SPLCALinear(embed_dim, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, num_classes)
        self.splca_layers = [self.fc1, self.fc2]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token ids
        x = self.embedding(x)  # (batch, seq, embed)
        x = x.mean(dim=1)  # simple pooling
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_splca_updates(self) -> list:
        '''Collect update dicts from all SPLCA layers'''
        updates = []
        
        # Compute predictions and errors
        for i, layer in enumerate(self.splca_layers[:-1]):
            next_layer = self.splca_layers[i + 1]
            next_output = next_layer.current_output
            if next_output is not None:
                error = layer.compute_local_error(next_output)
                updates.append(layer.get_update_dict(error))
        
        # Last layer uses output error (if available via external signal)
        return updates