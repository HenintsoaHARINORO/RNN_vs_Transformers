import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict, Tuple
import numpy as np

from pipeline2 import DataPipeline


class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self, num_classes=10, dropout=0.1):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class MNISTTrainer:
    """Trainer class for MNIST classification model"""

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            device: str = 'mps'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")

            for batch in progress_bar:
                # Move batch to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{current_acc:.2f}%'
                })

        avg_loss = total_loss / len(self.val_dataloader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """Train the model for specified epochs"""
        print(f"Starting training for {num_epochs} epochs")

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {val_acc:.2f}%. Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, save_path)

        return self.history


def plot_training_history(history: Dict, save_path: str = None):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)

    # Accuracy difference
    acc_diff = np.array(history['val_acc']) - np.array(history['train_acc'])
    ax4.plot(epochs, acc_diff, 'purple')
    ax4.set_title('Validation - Training Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")

    plt.show()


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
    Train CNN on MNIST dataset with specified perturbation.

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
    # Auto-detect device if not specified, with MPS preference for Mac
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
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
        'data_root': './data',
        'device': device
    }

    print(f"Setting up MNIST pipeline with {perturbation_type} perturbation")

    # Initialize data pipeline
    pipeline = DataPipeline(config)
    pipeline.setup_mnist(['train', 'test'])

    train_loader = pipeline.get_dataloader('mnist_train')
    val_loader = pipeline.get_dataloader('mnist_test')

    print(f"Training samples: {len(pipeline.get_dataset('mnist_train')):,}")
    print(f"Test samples: {len(pipeline.get_dataset('mnist_test')):,}")

    # Initialize model
    model = MNISTNet(num_classes=10, dropout=dropout).to(device)

    # For MPS, ensure model parameters are in the correct dtype
    if device == 'mps':
        model = model.to(torch.float32)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Initialize trainer
    trainer = MNISTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        device=device
    )

    # Train the model
    print(f"\nStarting training with {perturbation_type} perturbation")
    history = trainer.train(
        num_epochs=num_epochs,
        save_path=checkpoint_path
    )

    # Plot training history
    plot_training_history(history, save_path=plot_path)

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
            'dropout': dropout,
            'num_params': num_params
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


def demo_mnist_training():
    """Demonstrate training with different perturbation types"""
    perturbation_configs = [
        ('remove_random', {'p': 0.15}),
        ('replace_random', {'p': 0.1}),
        ('block_shuffle', {'k': 4}),
        ('plug_random', {'p': 0.1})
    ]

    for perturbation_type, perturbation_params in perturbation_configs:
        print(f"\n=== Training MNIST with {perturbation_type} perturbation ===")
        try:
            history = train_mnist_with_perturbation(
                perturbation_type=perturbation_type,
                perturbation_prob=0.5,  # Apply perturbation 50% of the time
                perturbation_params=perturbation_params,
                batch_size=128,
                num_epochs=10,  # Reduced for demo
                learning_rate=1e-3,
                dropout=0.1,
                save_dir='./checkpoints_mnist_demo',
                device='mps'  # Explicitly set to MPS for Mac
            )
        except Exception as e:
            print(f"Training with {perturbation_type} failed: {e}")


def test_perturbation_effects():
    """Test and visualize the effects of different perturbations"""
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    # Create save directory
    save_dir = './perturbation_visualizations'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define all perturbation types and their parameters
    perturbation_configs = {
        'remove_random': {'p': 0.3},
        'replace_random': {'p': 0.25},
        #'block_shuffle': {'k': 4},
        #'plug_random': {'p': 0.2}
    }

    num_samples = 6

    for perturbation_type, perturbation_params in perturbation_configs.items():
        print(f"Generating visualization for {perturbation_type} perturbation...")

        config = {
            'batch_size': num_samples,
            'perturbation_type': perturbation_type,
            'perturbation_prob': 1.0,  # Always apply for visualization
            'perturbation_params': perturbation_params,
            'data_root': './data'
        }

        pipeline = DataPipeline(config)
        pipeline.setup_mnist(['train'])
        dataset = pipeline.get_dataset('mnist_train')

        # Get samples from different classes
        samples = []
        classes_found = set()
        i = 0
        while len(samples) < num_samples and i < len(dataset):
            sample = dataset[i]
            label = sample['label']

            # Try to get diverse classes
            if len(classes_found) < num_samples and label not in classes_found:
                samples.append(sample)
                classes_found.add(label)
            elif len(samples) < num_samples:
                samples.append(sample)
            i += 1

        # Create visualization with 2 rows: original and perturbed
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

        for i, sample in enumerate(samples):
            # Original image
            original_img = sample['original_image'].squeeze().cpu().numpy()
            axes[0, i].imshow(original_img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original\nLabel: {sample["label"]}', fontsize=10)
            axes[0, i].axis('off')

            # Perturbed image
            perturbed_img = sample['image'].squeeze().cpu().numpy()
            axes[1, i].imshow(perturbed_img, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'{perturbation_type.replace("_", " ").title()}', fontsize=10)
            axes[1, i].axis('off')

        # Add main title with parameters
        param_str = ', '.join([f'{k}={v}' for k, v in perturbation_params.items()])
        plt.suptitle(f'MNIST {perturbation_type.replace("_", " ").title()} Perturbation\nParameters: {param_str}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save the visualization
        save_path = os.path.join(save_dir, f'mnist_{perturbation_type}_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

        plt.show()
        plt.close()

    print(f"\nAll perturbation visualizations saved to: {save_dir}")


if __name__ == "__main__":
    # Test perturbation visualization
    print("=== Testing perturbation effects ===")
    test_perturbation_effects()

    # Run demonstration of MNIST training
    print("\n=== Training MNIST with perturbations ===")
    demo_mnist_training()
