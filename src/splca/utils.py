# ============================================================================
# splca/utils.py - STABILIZED
# ============================================================================
import torch
import matplotlib.pyplot as plt
from typing import List, Dict


def train_splca_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    modulator,
    epochs: int = 10,
    device: str = 'cpu',
    grad_accum_steps: int = 1,  # Gradient accumulation steps
    weight_decay: float = 1e-4,  # Regularization
):
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'modulation': []}

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Clear gradients at the start of each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Forward pass
            output = model(data)

            # Skip NaN/Inf outputs
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Warning: NaN/Inf in output at batch {batch_idx}, skipping")
                continue

            loss = criterion(output, target)
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Apply weight decay manually
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.add_(weight_decay * param)

                # Get SPLCA updates
                layer_updates = model.get_splca_updates()

                # Apply SPLCA update
                optimizer.step(layer_updates)  # Pass layer updates to optimizer
                optimizer.zero_grad()  # Clear gradients

            train_loss += loss.item() * grad_accum_steps  # Scale back accumulated loss

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)

                if torch.isnan(output).any() or torch.isinf(output).any():
                    continue

                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        m = modulator(val_loss)
        optimizer.set_modulation(m)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['modulation'].append(m)

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Modulation: {m:.3f}')

    return history


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    
    axes[0, 1].plot(history['val_acc'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    
    axes[1, 0].plot(history['modulation'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('m(t)')
    axes[1, 0].set_title('Global Modulatory Scalar')
    
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
