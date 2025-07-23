import torch
import json
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict
import torch.nn.functional as F

from pipeline2 import DataPipeline


class MNISTClassifier(nn.Module):
    """Simple CNN classifier for MNIST with perturbations"""

    def __init__(self, num_classes=10, dropout=0.1):
        super(MNISTClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7

        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MNISTTrainer:
    """Trainer class for MNIST classification"""

    def __init__(self, model, train_dataloader, val_dataloader,
                 learning_rate=1e-3, weight_decay=1e-5, device='cpu'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Move model to device
        self.model.to(device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_dataloader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs, save_path=None):
        """Train the model for multiple epochs"""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'history': history
                    }, save_path)
                    print(f'Best model saved with validation accuracy: {val_acc:.2f}%')

        return history


def plot_training_history(history, save_path=None, title_suffix=""):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_title(f'Training and Validation Loss{title_suffix}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
    ax2.set_title(f'Training and Validation Accuracy{title_suffix}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_comparison_results(all_histories, save_dir):
    """Plot comparison of all perturbation types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']

    # Plot validation loss comparison
    for i, (perturbation_type, history) in enumerate(all_histories.items()):
        ax1.plot(history['val_loss'], label=f'{perturbation_type}',
                 color=colors[i], marker=markers[i], linewidth=2, markersize=4)

    ax1.set_title('Validation Loss Comparison Across Perturbation Types')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot validation accuracy comparison
    for i, (perturbation_type, history) in enumerate(all_histories.items()):
        ax2.plot(history['val_acc'], label=f'{perturbation_type}',
                 color=colors[i], marker=markers[i], linewidth=2, markersize=4)

    ax2.set_title('Validation Accuracy Comparison Across Perturbation Types')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(save_dir, f'perturbation_comparison_{timestamp}.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_path}")

    plt.show()

    return comparison_path


def train_mnist_with_perturbation(
        perturbation_type: str = 'remove_random',
        perturbation_prob: float = 0.1,
        perturbation_params: Dict = None,
        batch_size: int = 64,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        save_dir: str = './checkpoints',
        device: str = None  # Will auto-detect device
) -> Dict:
    """
    Train CNN classifier on MNIST dataset with specified perturbation.

    Args:
        perturbation_type: Type of perturbation ('remove_random', 'replace_random', 'block_shuffle', 'plug_random')
        perturbation_prob: Probability of applying perturbation
        perturbation_params: Parameters for the perturbation method
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        dropout: Dropout probability
        save_dir: Directory to save checkpoints and plots
        device: Device to run training on ('cuda', 'mps', 'cpu', or None for auto-detection)

    Returns:
        Training history dictionary
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Ensure MPS is built and available if selected
    if device == 'mps' and not torch.backends.mps.is_built():
        raise RuntimeError("MPS backend is not available. Ensure PyTorch is installed with MPS support.")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f'mnist_{perturbation_type}_{timestamp}.pt')
    plot_path = os.path.join(save_dir, f'training_curves_mnist_{perturbation_type}_{timestamp}.png')

    # Setup configuration
    config = {
        'batch_size': batch_size,
        'num_workers': 0,  # Set to 0 for MPS compatibility
        'perturbation_type': perturbation_type,
        'perturbation_prob': perturbation_prob,
        'perturbation_params': perturbation_params or {'p': 0.1},
        'data_root': './data'
    }

    print(f"Setting up data pipeline with {perturbation_type} perturbation")

    # Initialize data pipeline
    pipeline = DataPipeline(config)
    pipeline.setup_mnist(['train', 'test'])

    train_loader = pipeline.get_dataloader('mnist_train')
    val_loader = pipeline.get_dataloader('mnist_test')

    print(f"Training samples: {len(pipeline.get_dataset('mnist_train')):,}")
    print(f"Test samples: {len(pipeline.get_dataset('mnist_test')):,}")

    # Initialize model
    model = MNISTClassifier(num_classes=10, dropout=dropout)

    # For MPS, ensure model parameters are in the correct dtype
    if device == 'mps':
        model = model.to(torch.float32)

    # Initialize trainer
    trainer = MNISTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        device=device
    )

    # Train the model
    print(f"\nStarting training with {perturbation_type} perturbation")
    history = trainer.train(
        num_epochs=num_epochs,
        save_path=checkpoint_path
    )

    # Plot training history with perturbation type in title
    plot_training_history(history, save_path=plot_path, title_suffix=f" ({perturbation_type})")

    # Save final training metrics
    metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'perturbation_type': perturbation_type,
        'perturbation_prob': perturbation_prob,
        'model_config': {
            'num_classes': 10,
            'dropout': dropout
        }
    }

    metrics_path = os.path.join(save_dir, f'metrics_mnist_{perturbation_type}_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTraining completed. Metrics saved to {metrics_path}")
    print(f"Final training accuracy: {metrics['final_train_acc']:.2f}%")
    print(f"Final validation accuracy: {metrics['final_val_acc']:.2f}%")
    print(f"Best validation accuracy: {metrics['best_val_acc']:.2f}%")

    return history


def demo_training():
    """Demonstrate training with all 4 perturbation types and save comparison visualization"""
    perturbation_configs = [
        ('remove_random', {'p': 0.15}),
        ('replace_random', {'p': 0.1}),
        #('block_shuffle', {'k': 4}),
        #('plug_random', {'p': 0.1})
    ]

    save_dir = './checkpoints_mnist_demo'
    os.makedirs(save_dir, exist_ok=True)

    all_histories = {}
    all_metrics = {}

    for perturbation_type, perturbation_params in perturbation_configs:
        print(f"\n=== Training MNIST with {perturbation_type} perturbation ===")
        try:
            history = train_mnist_with_perturbation(
                perturbation_type=perturbation_type,
                perturbation_prob=0.5,  # Apply perturbation 50% of the time
                perturbation_params=perturbation_params,
                batch_size=64,
                num_epochs=10,  # Reduced for demo
                learning_rate=1e-3,
                dropout=0.1,
                save_dir=save_dir,
                device='mps'  # Explicitly set to MPS for Mac
            )

            all_histories[perturbation_type] = history
            all_metrics[perturbation_type] = {
                'final_val_acc': history['val_acc'][-1],
                'best_val_acc': max(history['val_acc']),
                'final_train_acc': history['train_acc'][-1],
                'best_train_acc': max(history['train_acc'])
            }

        except Exception as e:
            print(f"Training with {perturbation_type} failed: {e}")

    # Create comparison visualization
    if len(all_histories) > 1:
        print(f"\n=== Creating comparison visualization for {len(all_histories)} perturbation types ===")
        comparison_path = plot_comparison_results(all_histories, save_dir)

        # Save summary metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(save_dir, f'summary_metrics_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        print(f"\nSummary metrics saved to {summary_path}")

        # Print comparison summary
        print("\n=== Performance Summary ===")
        print(
            f"{'Perturbation Type':<15} {'Final Val Acc':<12} {'Best Val Acc':<12} {'Final Train Acc':<15} {'Best Train Acc':<15}")
        print("-" * 75)
        for perturbation_type, metrics in all_metrics.items():
            print(f"{perturbation_type:<15} {metrics['final_val_acc']:<12.2f} {metrics['best_val_acc']:<12.2f} "
                  f"{metrics['final_train_acc']:<15.2f} {metrics['best_train_acc']:<15.2f}")

    return all_histories, all_metrics


if __name__ == "__main__":
    # Run
    all_histories, all_metrics = demo_training()
