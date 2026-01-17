# ============================================================================
# examples/text_classification_demo.py
# ============================================================================
"""
SPLCA Text Classification Demo
Simple sentiment analysis using a simple synthetic dataset
Run with: python examples/text_classification_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.splca import SPLCAOptimizer, TextClassifier, ValidationModulator
from src.splca.utils import train_splca_model, plot_training_history


class SimpleTextDataset(Dataset):
    """Simple synthetic text classification dataset"""
    def __init__(self, num_samples=10000, vocab_size=5000, max_len=100, num_classes=4):
        self.data = []
        for _ in range(num_samples):
            # Generate random sequences
            seq_len = torch.randint(10, max_len, (1,)).item()
            token_ids = torch.randint(0, vocab_size, (seq_len,))
            label = torch.randint(0, num_classes, (1,)).item()
            self.data.append((token_ids, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token_ids, label = self.data[idx]
        return token_ids, torch.tensor(label, dtype=torch.long)


def collate_batch(batch):
    """Collate function to pad sequences"""
    texts, labels = zip(*batch)
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels


def main():
    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 4
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    MAX_LEN = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')

    # Create synthetic datasets
    print("Creating datasets...")
    train_dataset = SimpleTextDataset(
        num_samples=10000, 
        vocab_size=VOCAB_SIZE, 
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES
    )
    test_dataset = SimpleTextDataset(
        num_samples=2000, 
        vocab_size=VOCAB_SIZE, 
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES
    )
    
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
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