import torch
import json
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict, Tuple
import numpy as np


from GPT import GPT, GPTConfig
from pipeline2 import DataPipeline


class GPTTrainer:
    """Trainer class for GPT language model"""

    def __init__(
            self,
            model: GPT,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            learning_rate: float = 3e-4,
            weight_decay: float = 1e-1,
            grad_clip_norm: float = 1.0,
            device: str = 'mps'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.grad_clip_norm = grad_clip_norm

        # Configure optimizer using the model's built-in method
        # For MPS, use 'cuda' as device_type since it's optimized for that path
        device_type = 'cuda' if device == 'mps' else device
        self.optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(0.9, 0.95),
            device_type=device_type
        )

        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'learning_rates': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
            else:
                input_ids = batch.to(self.device)

            # Create targets (shift input_ids by one position)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            # Forward pass
            logits, loss = self.model(input_ids, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{math.exp(avg_loss):.2f}'
            })

        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)

        return avg_loss, avg_ppl

    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")

            for batch in progress_bar:
                # Move batch to device
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Create targets
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

                # Forward pass
                logits, loss = self.model(input_ids, targets)

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'val_loss': f'{avg_loss:.4f}',
                    'val_ppl': f'{math.exp(avg_loss):.2f}'
                })

        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)

        return avg_loss, avg_ppl

    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """Train the model for specified epochs"""
        print(f"Starting training for {num_epochs} epochs")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_ppl = self.train_epoch()

            # Validate
            val_loss, val_ppl = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}. Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.model.config
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

    # Perplexity curves
    ax2.plot(epochs, history['train_ppl'], 'b-', label='Train PPL')
    ax2.plot(epochs, history['val_ppl'], 'r-', label='Val PPL')
    ax2.set_title('Training and Validation Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True)

    # Learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)

    # Loss difference
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    ax4.plot(epochs, loss_diff, 'purple')
    ax4.set_title('Validation - Training Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")

    plt.show()


def train_gpt_with_perturbation(
        perturbation_type: str = 'remove_random',
        perturbation_prob: float = 0.1,
        perturbation_params: Dict = None,
        batch_size: int = 16,
        max_length: int = 256,
        num_epochs: int = 5,
        learning_rate: float = 3e-4,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,
        bias: bool = True,
        save_dir: str = './checkpoints',
        device: str = None  # Will auto-detect device
) -> Dict:
    """
    Train GPT language model on WikiText-103 dataset with specified perturbation.

    Args:
        perturbation_type: Type of perturbation ('remove_random', 'replace_random', 'block_shuffle', 'plug_random')
        perturbation_prob: Probability of applying perturbation
        perturbation_params: Parameters for the perturbation method
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
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
    checkpoint_path = os.path.join(save_dir, f'gpt_{perturbation_type}_{timestamp}.pt')
    plot_path = os.path.join(save_dir, f'training_curves_{perturbation_type}_{timestamp}.png')

    # Setup configuration
    config = {
        'batch_size': batch_size,
        'max_length': max_length,
        'tokenizer_type': 'gpt2',
        'num_workers': 0,  # Set to 0 for MPS compatibility
        'perturbation_type': perturbation_type,
        'perturbation_prob': perturbation_prob,
        'perturbation_params': perturbation_params or {'p': 0.1}
    }

    print(f"Setting up data pipeline with {perturbation_type} perturbation")

    # Initialize data pipeline
    pipeline = DataPipeline(config)
    pipeline.setup_wikitext(['train', 'validation'])

    train_loader = pipeline.get_dataloader('wikitext_train')
    val_loader = pipeline.get_dataloader('wikitext_validation')

    # Get vocabulary size from tokenizer
    vocab_size = pipeline.get_dataset('wikitext_train').tokenizer_wrapper.get_vocab_size()

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Training samples: {len(pipeline.get_dataset('wikitext_train')):,}")
    print(f"Validation samples: {len(pipeline.get_dataset('wikitext_validation')):,}")

    # Initialize GPT model configuration
    gpt_config = GPTConfig(
        block_size=max_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias
    )

    # Initialize model
    model = GPT(gpt_config).to(device)

    # For MPS, ensure model parameters are in the correct dtype
    if device == 'mps':
        model = model.to(torch.float32)

    print(f"Model parameters: {model.get_num_params():,}")

    # Initialize trainer
    trainer = GPTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        weight_decay=1e-1,
        grad_clip_norm=1.0,
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
        'final_train_ppl': history['train_ppl'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_ppl': history['val_ppl'][-1],
        'perturbation_type': perturbation_type,
        'perturbation_prob': perturbation_prob,
        'model_config': {
            'vocab_size': vocab_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'dropout': dropout,
            'bias': bias,
            'block_size': max_length
        }
    }

    metrics_path = os.path.join(save_dir, f'metrics_{perturbation_type}_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTraining completed. Metrics saved to {metrics_path}")
    print(f"Final training perplexity: {metrics['final_train_ppl']:.2f}")
    print(f"Final validation perplexity: {metrics['final_val_ppl']:.2f}")

    return history


def load_pretrained_gpt(
        model_type: str = 'gpt2',
        override_args: Dict = None,
        device: str = None
) -> GPT:
    """
    Load a pretrained GPT model and prepare it for training.

    Args:
        model_type: Type of GPT model ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        override_args: Optional arguments to override (e.g., {'dropout': 0.1})
        device: Device to load model on

    Returns:
        Loaded GPT model
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Loading pretrained {model_type} model...")
    model = GPT.from_pretrained(model_type, override_args=override_args)
    model = model.to(device)

    if device == 'mps':
        model = model.to(torch.float32)

    print(f"Model loaded on {device}")
    return model


def demo_training():
    """Demonstrate training with different perturbation types"""
    perturbation_configs = [
        ('remove_random', {'p': 0.15}),
        ('replace_random', {'p': 0.1}),
        ('block_shuffle', {'k': 4}),
        ('plug_random', {'p': 0.1})
    ]

    for perturbation_type, perturbation_params in perturbation_configs:
        print(f"\n=== Training with {perturbation_type} perturbation ===")
        try:
            history = train_gpt_with_perturbation(
                perturbation_type=perturbation_type,
                perturbation_prob=0.5,  # Apply perturbation 50% of the time
                perturbation_params=perturbation_params,
                batch_size=8,  # Reduced for MPS
                max_length=256,
                num_epochs=10,  # Reduced for demo
                learning_rate=6e-5,  # Lower learning rate for GPT
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.1,
                save_dir='./checkpoints_demo',
                device='mps'  # Explicitly set to MPS for Mac
            )
        except Exception as e:
            print(f"Training with {perturbation_type} failed: {e}")


def demo_pretrained_training():
    """Demonstrate fine-tuning a pretrained GPT model"""
    print("\n=== Fine-tuning pretrained GPT-2 ===")

    # Load pretrained model
    model = load_pretrained_gpt('gpt2', override_args={'dropout': 0.1}, device='mps')

    # Setup data pipeline (same as before)
    config = {
        'batch_size': 8,
        'max_length': 512,  # GPT-2 can handle longer sequences
        'tokenizer_type': 'gpt2',
        'num_workers': 0,
        'perturbation_type': 'remove_random',
        'perturbation_prob': 0.3,
        'perturbation_params': {'p': 0.1}
    }

    pipeline = DataPipeline(config)
    pipeline.setup_wikitext(['train', 'validation'])

    train_loader = pipeline.get_dataloader('wikitext_train')
    val_loader = pipeline.get_dataloader('wikitext_validation')

    # Initialize trainer with lower learning rate for fine-tuning
    trainer = GPTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=1e-5,  # Much lower for fine-tuning
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        device='mps'
    )

    # Train
    history = trainer.train(
        num_epochs=2,
        save_path='./checkpoints_demo/gpt2_finetuned.pt'
    )

    plot_training_history(history, save_path='./checkpoints_demo/gpt2_finetuned_curves.png')

    print("Fine-tuning completed!")


if __name__ == "__main__":
    # Run demonstration of training from scratch
    print("=== Training GPT from scratch ===")
    demo_training()

    # Run demonstration of fine-tuning pretrained model
    print("\n=== Fine-tuning pretrained GPT-2 ===")
    demo_pretrained_training()

    print("\n=== Usage Examples ===")
    print("""
    # Train from scratch with specific perturbation
    history = train_gpt_with_perturbation(
        perturbation_type='remove_random',
        perturbation_prob=0.3,
        perturbation_params={'p': 0.15},
        batch_size=16,
        max_length=512,
        num_epochs=10,
        learning_rate=3e-4,
        n_layer=12,
        n_head=12,
        n_embd=768,
        save_dir='./checkpoints',
        device='mps'  # Use MPS for Mac
    )

    # Load pretrained model for fine-tuning
    model = load_pretrained_gpt('gpt2', override_args={'dropout': 0.1}, device='mps')

    # Access training results
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation perplexity: {history['val_ppl'][-1]:.2f}")
    """)