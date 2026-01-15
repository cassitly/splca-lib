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
):
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'modulation': []}

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Forward pass
            output = model(data)

            # Skip NaN/Inf outputs
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Warning: NaN/Inf in output at batch {batch_idx}, skipping")
                continue

            loss = criterion(output, target)

            # Get SPLCA updates
            updates = model.get_splca_updates()

            # Apply SPLCA update
            optimizer.step(updates)

            # Optimize predictor updates
            for layer in model.splca_layers:
                if hasattr(layer, 'predictor') and layer.prev_output is not None:
                    pred_out = layer.predict_next()
                    if pred_out is not None and layer.current_output is not None:
                        pred_out_clipped = torch.clamp(pred_out, -10.0, 10.0)
                        target_clipped = torch.clamp(layer.current_output.detach(), -10.0, 10.0)

                        # Ensure sizes match
                        if pred_out_clipped.size() != target_clipped.size():
                            min_size = [
                                min(pred_out_clipped.size(i), target_clipped.size(i))
                                for i in range(len(pred_out_clipped.size()))
                            ]
                            pred_out_clipped = pred_out_clipped[tuple(slice(0, s) for s in min_size)]
                            target_clipped = target_clipped[tuple(slice(0, s) for s in min_size)]

                        pred_loss = ((pred_out_clipped - target_clipped)**2).mean()

                        if not torch.isnan(pred_loss) and not torch.isinf(pred_loss) and pred_loss < 1000.0:
                            pred_loss.backward()
                            for p in layer.predictor.parameters():
                                if p.grad is not None:
                                    # Simplified gradient update
                                    p.data.add_(-optimizer.eta_pred, p.grad)
                                    p.grad.zero_()

            train_loss += loss.item()

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
