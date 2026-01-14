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


class SimpleTextDataset(Dataset):
    '''Dummy text dataset for demo'''
    def __init__(self, num_samples=1000, vocab_size=5000, max_length=50):
        self.data = torch.randint(0, vocab_size, (num_samples, max_length))
        self.labels = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # Create dummy datasets
    train_dataset = SimpleTextDataset(num_samples=2000)
    test_dataset = SimpleTextDataset(num_samples=500)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
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