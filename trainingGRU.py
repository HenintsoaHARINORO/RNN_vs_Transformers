import torch
import json
from datetime import datetime
import os
from typing import Dict
from pipeline2 import DataPipeline
from GRU import GRULanguageModel, GRUTrainer, plot_training_history


def train_gru_with_perturbation(
        perturbation_type: str = 'remove_random',
        perturbation_prob: float = 0.1,
        perturbation_params: Dict = None,
        batch_size: int = 16,
        max_length: int = 256,
        num_epochs: int = 5,
        learning_rate: float = 1e-3,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        save_dir: str = './checkpoints',
        device: str = None  # Will auto-detect device
) -> Dict:
    """
    Train GRU language model on WikiText-103 dataset with specified perturbation.

    Args:
        perturbation_type: Type of perturbation ('remove_random', 'replace_random', 'block_shuffle', 'plug_random')
        perturbation_prob: Probability of applying perturbation
        perturbation_params: Parameters for the perturbation method
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        embed_dim: Embedding dimension
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
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
    checkpoint_path = os.path.join(save_dir, f'gru_{perturbation_type}_{timestamp}.pt')
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

    # Initialize model
    model = GRULanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=True,
        use_positional_encoding=True,
        pad_token_id=0
    ).to(device)

    # For MPS, ensure model parameters are in the correct dtype
    if device == 'mps':
        model = model.to(torch.float32)  # MPS typically requires float32

    # Initialize trainer
    trainer = GRUTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        weight_decay=1e-5,
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
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
    }

    metrics_path = os.path.join(save_dir, f'metrics_{perturbation_type}_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTraining completed. Metrics saved to {metrics_path}")
    print(f"Final training perplexity: {metrics['final_train_ppl']:.2f}")
    print(f"Final validation perplexity: {metrics['final_val_ppl']:.2f}")

    return history


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
            history = train_gru_with_perturbation(
                perturbation_type=perturbation_type,
                perturbation_prob=0.5,  # Apply perturbation 50% of the time
                perturbation_params=perturbation_params,
                batch_size=16,
                max_length=256,
                num_epochs=10,  # Reduced for demo
                learning_rate=1e-3,
                embed_dim=256,
                hidden_dim=512,
                num_layers=2,
                dropout=0.1,
                save_dir='./checkpoints_demo',
                device='mps'  # Explicitly set to MPS for Mac
            )
        except Exception as e:
            print(f"Training with {perturbation_type} failed: {e}")


if __name__ == "__main__":
    # Run demonstration
    demo_training()

    print("\n=== Usage Example ===")
    print("""
    # Train with specific perturbation
    history = train_gru_with_perturbation(
        perturbation_type='remove_random',
        perturbation_prob=0.3,
        perturbation_params={'p': 0.15},
        batch_size=32,
        max_length=512,
        num_epochs=10,
        learning_rate=1e-3,
        save_dir='./checkpoints',
        device='mps'  # Use MPS for Mac
    )

    # Access training results
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation perplexity: {history['val_ppl'][-1]:.2f}")
    """)