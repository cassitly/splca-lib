# ============================================================================
# splca/models/text.py
# ============================================================================
import torch
import torch.nn as nn
from ..layers import SPLCALinear
from ..modulation import SPLCAModulator


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
        self.modulator = SPLCAModulator(mode='validation')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
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
        for layer in self.splca_layers:
            local_error = layer.compute_local_error()
            if local_error is not None and layer.current_input is not None:
                # Aggregate over batch dimension
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

    @staticmethod
    def preprocess_text(texts, tokenizer, max_length):
        '''
        Tokenize and pad text inputs.
        '''
        tokenized = [tokenizer(text) for text in texts]
        padded = [tokens[:max_length] + [0] * (max_length - len(tokens)) for tokens in tokenized]
        return torch.tensor(padded, dtype=torch.long)
