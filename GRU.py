import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Tuple, Optional
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


class PositionalEncoding(nn.Module):
    """Optional positional encoding for sequence modeling"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GRULanguageModel(nn.Module):
    """GRU-based Language Model for next-token prediction"""

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 2,
            dropout: float = 0.1,
            tie_weights: bool = True,
            use_positional_encoding: bool = False,
            pad_token_id: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # Optional positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim)

        # Dropout for embedding
        self.embed_dropout = nn.Dropout(dropout)

        # GRU layers
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Tie embedding and output weights for parameter efficiency
        if tie_weights:
            if embed_dim != hidden_dim:
                print("Warning: embed_dim != hidden_dim, adding projection layer for weight tying")
                self.tie_projection = nn.Linear(hidden_dim, embed_dim)
                self.output_projection.weight = self.embedding.weight
            else:
                self.output_projection.weight = self.embedding.weight

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output dropout
        self.output_dropout = nn.Dropout(dropout)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'tie_projection'):
            self.tie_projection.weight.data.uniform_(-initrange, initrange)
        else:
            self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
            self,
            input_ids: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
            return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            hidden: Initial hidden state (num_layers, batch_size, hidden_dim)
            return_hidden: Whether to return final hidden state

        Returns:
            Dictionary containing logits and optionally hidden state
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)

        # Apply embedding dropout
        embeddings = self.embed_dropout(embeddings)

        # GRU forward pass
        gru_output, hidden_final = self.gru(embeddings, hidden)
        # gru_output: (batch_size, seq_len, hidden_dim)
        # hidden_final: (num_layers, batch_size, hidden_dim)

        # Apply layer normalization and dropout
        gru_output = self.layer_norm(gru_output)
        gru_output = self.output_dropout(gru_output)

        # Project to vocabulary
        if hasattr(self, 'tie_projection'):
            gru_output = self.tie_projection(gru_output)

        logits = self.output_projection(gru_output)  # (batch_size, seq_len, vocab_size)

        result = {'logits': logits}
        if return_hidden:
            result['hidden'] = hidden_final

        return result

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the trained model

        Args:
            input_ids: Starting sequence (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            pad_token_id: Padding token ID

        Returns:
            Generated sequences (batch_size, max_length)
        """
        self.eval()
        pad_token_id = pad_token_id or self.pad_token_id

        with torch.no_grad():
            batch_size = input_ids.shape[0]
            device = input_ids.device

            # Initialize generation
            generated = input_ids.clone()
            hidden = None

            for _ in range(max_length - input_ids.shape[1]):
                # Get logits for the last token
                outputs = self.forward(generated, hidden, return_hidden=True)
                logits = outputs['logits'][:, -1, :]  # (batch_size, vocab_size)
                hidden = outputs['hidden']

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    topk_logits, topk_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, topk_indices, topk_logits)

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Stop if all sequences hit pad token (optional)
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break

            return generated


class GRUTrainer:
    """Trainer class for GRU Language Model"""

    def __init__(
            self,
            model: GRULanguageModel,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            grad_clip_norm: float = 1.0,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.grad_clip_norm = grad_clip_norm

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        # Training history
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
        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs['logits']

            # Calculate loss
            # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            batch_tokens = (labels != self.model.pad_token_id).sum().item()
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

            # Update progress bar
            current_ppl = math.exp(batch_loss)
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'ppl': f'{current_ppl:.2f}'
            })

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        return avg_loss, perplexity

    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return float('inf'), float('inf')

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits']

                # Calculate loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                # Update metrics
                batch_loss = loss.item()
                batch_tokens = (labels != self.model.pad_token_id).sum().item()
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        return avg_loss, perplexity

    def train(self, num_epochs: int, save_path: Optional[str] = None) -> Dict:
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_ppl = self.train_epoch()

            # Validate
            val_loss, val_ppl = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rates'].append(current_lr)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            print(f"Learning Rate: {current_lr:.2e}")

            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_path, epoch, is_best=True)
                print(f"New best model saved to {save_path}")

        return self.history

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'pad_token_id': self.model.pad_token_id
            }
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

        return checkpoint['epoch']


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if history['val_loss'] and not all(x == float('inf') for x in history['val_loss']):
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Perplexity plot
    ax2.plot(epochs, history['train_ppl'], 'b-', label='Train PPL')
    if history['val_ppl'] and not all(x == float('inf') for x in history['val_ppl']):
        ax2.plot(epochs, history['val_ppl'], 'r-', label='Validation PPL')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True)

    # Learning rate plot
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True)

    # Combined loss and LR
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    line2 = ax4_twin.plot(epochs, history['learning_rates'], 'g--', label='Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Learning Rate', color='g')
    ax4_twin.set_yscale('log')
    ax4.set_title('Loss vs Learning Rate')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def demo_gru_model():
    """Demonstration of GRU model"""
    print("=== GRU Language Model Demo ===\n")

    # Mock data for demonstration
    vocab_size = 1000
    batch_size = 8
    seq_len = 32

    # Create sample data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Sample input shape: {input_ids.shape}")
    print(f"Sample labels shape: {labels.shape}")

    # Initialize model
    model = GRULanguageModel(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        tie_weights=True,
        use_positional_encoding=False
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']

        print(f"Output logits shape: {logits.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")

        # Test loss calculation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        perplexity = torch.exp(loss)

        print(f"Sample loss: {loss.item():.4f}")
        print(f"Sample perplexity: {perplexity.item():.2f}")

    # Test generation
    print("\n--- Testing Generation ---")
    prompt = torch.randint(0, vocab_size, (1, 10))  # Single sequence prompt

    generated = model.generate(
        prompt,
        max_length=20,
        temperature=0.8,
        top_k=50
    )

    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")

