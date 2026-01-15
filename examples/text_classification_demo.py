# ============================================================================
# examples/text_classification_demo.py
# ============================================================================
"""
SPLCA Text Classification Demo
Simple sentiment analysis on dummy data
Run with: python examples/text_classification_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from splca import SPLCAOptimizer, TextClassifier, ValidationModulator
from splca.utils import train_splca_model, plot_training_history
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class AGNewsDataset(Dataset):
    """AG News Dataset for text classification"""
    def __init__(self, data_iter):
        tokenizer = get_tokenizer("basic_english")
        
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        
        vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        def text_pipeline(x):
            return vocab(tokenizer(x))
        
        def label_pipeline(x):
            return int(x) - 1
        
        self.data = [(text_pipeline(text), label_pipeline(label)) for label, text in data_iter]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return torch.tensor(text), torch.tensor(label)


def main():
    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    BATCH_SIZE = 64
    EPOCHS = 95
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # Use a realistic text dataset
    train_iter, test_iter = AG_NEWS(split=("train", "test"))
    
    train_dataset = AGNewsDataset(train_iter)
    test_dataset = AGNewsDataset(test_iter)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: zip(*x))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: zip(*x))
    
    # Model
    model = TextClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )
    
    # SPLCA optimizer
    optimizer = SPLCAOptimizer(
        model.parameters(),
        lr=LEARNING_RATE,
        gamma=0.95
    )
    
    # Modulator
    modulator = ValidationModulator(alpha=1.0)
    
    # Train
    print('\nStarting SPLCA training on text classification...')
    history = train_splca_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        modulator=modulator,
        epochs=EPOCHS,
        device=DEVICE
    )
    
    # Plot results
    plot_training_history(history, save_path='text_splca_results.png')
    
    print(f'\nFinal validation accuracy: {history["val_acc"][-1]:.4f}')


if __name__ == '__main__':
    main()