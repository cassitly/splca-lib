# ============================================================================
# splca/models/text.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear


class TextClassifier(nn.Module):
    '''
    Enhanced text classification model using SPLCA layers.
    Embeddings → BiLSTM → SPLCA Linear → SPLCA Linear → Output
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
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc1 = SPLCALinear(hidden_dim * 2, hidden_dim)
        self.fc2 = SPLCALinear(hidden_dim, num_classes)
        self.splca_layers = [self.fc1, self.fc2]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)
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
