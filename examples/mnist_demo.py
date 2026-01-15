# ============================================================================
# examples/mnist_demo.py
# ============================================================================
"""
SPLCA MNIST Demo
Run with: python examples/mnist_demo.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from splca import SPLCAOptimizer, VisionClassifier, ValidationModulator
from splca.utils import train_splca_model, plot_training_history


def main():
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    GAMMA = 0.95
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Model
    model = VisionClassifier(
        in_channels=1,
        num_classes=10,
        hidden_dim=128
    )
    
    # SPLCA optimizer
    optimizer = SPLCAOptimizer(
        model.parameters(),
        lr=LEARNING_RATE,
        gamma=GAMMA,
        eta_pred=1e-3,
        eta_heb=1e-4,
        eta_stab=1e-5
    )
    
    # Modulator
    modulator = ValidationModulator(alpha=1.0)
    
    # Train
    print('Starting SPLCA training on MNIST...')
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
    plot_training_history(history, save_path='mnist_splca_results.png')
    
    print(f'Final validation accuracy: {history["val_acc"][-1]:.4f}')
    print('Results saved to mnist_splca_results.png')


if __name__ == '__main__':
    main()