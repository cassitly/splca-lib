# ============================================================================
# examples/audio_classification_demo.py
# ============================================================================
"""
SPLCA Audio Classification Demo
Run with: python examples/audio_classification_demo.py
"""

import torch
from torch.utils.data import Dataset, DataLoader

from splca import SPLCAOptimizer, AudioClassifier, ValidationModulator
from splca.utils import train_splca_model, plot_training_history


class SimpleAudioDataset(Dataset):
    '''Dummy audio dataset (e.g., mel spectrograms)'''
    def __init__(self, num_samples=1000, input_dim=128):
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    # Hyperparameters
    INPUT_DIM = 128  # e.g., 128 mel bins
    NUM_CLASSES = 10
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    EPOCHS = 115
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # Create datasets
    train_dataset = SimpleAudioDataset(num_samples=2000)
    test_dataset = SimpleAudioDataset(num_samples=500)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = AudioClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM
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
    print('\nStarting SPLCA training on audio classification...')
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
    plot_training_history(history, save_path='audio_splca_results.png')
    
    print(f'\nFinal validation accuracy: {history["val_acc"][-1]:.4f}')


if __name__ == '__main__':
    main()