import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union
from transformers import GPT2Tokenizer
import torchvision
from torchvision import transforms
from datasets import load_dataset
import torchvision.datasets as datasets
import math
from collections import Counter


class TokenizerWrapper:
    """Wrapper for handling tokenization with vocabulary management"""

    def __init__(self, tokenizer_type='gpt2', vocab_size=50000):
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size

        if tokenizer_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Simple word-based tokenizer for demonstration
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
            self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
            self.tokenizer = None

    def encode(self, text: str, max_length: int = 512) -> List[int]:
        if self.tokenizer_type == 'gpt2':
            encoded = self.tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            return encoded
        else:
            # Simple word tokenization
            words = text.lower().split()
            tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]

            # Pad or truncate
            if len(tokens) < max_length:
                tokens.extend([self.word_to_idx['<PAD>']] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]

            return tokens

    def decode(self, token_ids: List[int]) -> str:
        if self.tokenizer_type == 'gpt2':
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            words = [self.idx_to_word.get(idx, '<UNK>') for idx in token_ids]
            return ' '.join(words)

    def get_vocab_size(self) -> int:
        if self.tokenizer_type == 'gpt2':
            return len(self.tokenizer)
        else:
            return len(self.word_to_idx)


class TextPerturbations:
    """Collection of text perturbation methods"""

    @staticmethod
    def remove_random_tokens(tokens: List[int], p: float = 0.1, pad_token: int = 0) -> List[int]:
        """Randomly drop tokens with probability p"""
        perturbed = []
        for token in tokens:
            if token != pad_token and random.random() < p:
                continue  # Drop this token
            perturbed.append(token)

        # Pad to original length
        original_length = len(tokens)
        while len(perturbed) < original_length:
            perturbed.append(pad_token)

        return perturbed[:original_length]

    @staticmethod
    def replace_random_tokens(tokens: List[int], p: float = 0.1, vocab_size: int = 50000,
                              pad_token: int = 0) -> List[int]:
        """Replace percentage p of tokens with random tokens from vocabulary"""
        perturbed = tokens.copy()
        non_pad_indices = [i for i, token in enumerate(tokens) if token != pad_token]

        n_replace = int(len(non_pad_indices) * p)
        replace_indices = random.sample(non_pad_indices, min(n_replace, len(non_pad_indices)))

        for idx in replace_indices:
            # Sample random token (avoid special tokens)
            random_token = random.randint(4, vocab_size - 1)  # Assuming first 4 are special tokens
            perturbed[idx] = random_token

        return perturbed

    @staticmethod
    def block_shuffle(tokens: List[int], k: int = 4, pad_token: int = 0) -> List[int]:
        """Split input into k equal-length blocks and shuffle them"""
        # Find non-padding length
        non_pad_tokens = [token for token in tokens if token != pad_token]
        if len(non_pad_tokens) == 0:
            return tokens

        # Split into k blocks
        block_size = len(non_pad_tokens) // k
        if block_size == 0:
            return tokens

        blocks = []
        for i in range(k):
            start_idx = i * block_size
            if i == k - 1:  # Last block gets remaining tokens
                block = non_pad_tokens[start_idx:]
            else:
                block = non_pad_tokens[start_idx:start_idx + block_size]
            blocks.append(block)

        # Shuffle blocks
        random.shuffle(blocks)

        # Reconstruct sequence
        shuffled_tokens = []
        for block in blocks:
            shuffled_tokens.extend(block)

        # Pad to original length
        while len(shuffled_tokens) < len(tokens):
            shuffled_tokens.append(pad_token)

        return shuffled_tokens[:len(tokens)]

    @staticmethod
    def plug_random_tokens(tokens: List[int], p: float = 0.1, vocab_size: int = 50000,
                           max_length: int = 512, pad_token: int = 0) -> List[int]:
        """Insert random tokens at random positions, maintaining max length"""
        non_pad_tokens = [token for token in tokens if token != pad_token]
        if len(non_pad_tokens) == 0:
            return tokens

        n_insert = int(len(non_pad_tokens) * p)

        # Insert random tokens
        result = non_pad_tokens.copy()
        for _ in range(n_insert):
            if len(result) >= max_length:
                break
            insert_pos = random.randint(0, len(result))
            random_token = random.randint(4, vocab_size - 1)
            result.insert(insert_pos, random_token)

        # Truncate to max_length and pad
        result = result[:max_length]
        while len(result) < max_length:
            result.append(pad_token)

        return result


class ImagePerturbations:
    """Collection of image perturbation methods for MNIST"""

    @staticmethod
    def remove_random_pixels(image: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly set pixels to 0 with probability p"""
        mask = torch.rand_like(image) > p
        return image * mask.float()

    @staticmethod
    def replace_random_pixels(image: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Replace percentage p of pixels with random values"""
        mask = torch.rand_like(image) < p
        noise = torch.rand_like(image)
        return torch.where(mask, noise, image)

    @staticmethod
    def block_shuffle(image: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Split image into k×k blocks and shuffle them"""
        if len(image.shape) == 2:  # Single channel
            h, w = image.shape
        else:  # Multi-channel
            c, h, w = image.shape
            image = image.view(h, w)  # Assume single channel for MNIST

        block_h, block_w = h // k, w // k

        # Extract blocks
        blocks = []
        for i in range(k):
            for j in range(k):
                start_h, start_w = i * block_h, j * block_w
                block = image[start_h:start_h + block_h, start_w:start_w + block_w]
                blocks.append(block)

        # Shuffle blocks
        random.shuffle(blocks)

        # Reconstruct image
        result = torch.zeros_like(image)
        idx = 0
        for i in range(k):
            for j in range(k):
                start_h, start_w = i * block_h, j * block_w
                result[start_h:start_h + block_h, start_w:start_w + block_w] = blocks[idx]
                idx += 1

        return result.unsqueeze(0) if len(image.shape) == 3 else result

    @staticmethod
    def plug_random_pixels(image: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Add random noise to random positions"""
        mask = torch.rand_like(image) < p
        noise = torch.rand_like(image) * 0.5  # Lower intensity noise
        return image + mask.float() * noise


class WikiTextDataset(Dataset):
    """WikiText-103 dataset with perturbations"""

    def __init__(self, split='train', max_length=512, tokenizer_type='gpt2',
                 perturbation_type=None, perturbation_prob=0.1, perturbation_params=None,
                 max_samples=None):
        self.max_length = max_length
        self.perturbation_type = perturbation_type
        self.perturbation_prob = perturbation_prob
        self.perturbation_params = perturbation_params or {}

        # Initialize tokenizer
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_type)

        # Load WikiText-103
        try:
            full_dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)

            # Limit samples first
            if max_samples is not None:
                dataset_size = min(max_samples, len(full_dataset))
                self.dataset = full_dataset.select(range(dataset_size))
            else:
                self.dataset = full_dataset

            # Filter out empty texts from the limited dataset
            self.dataset = self.dataset.filter(lambda x: len(x['text'].strip()) > 10)

        except Exception as e:
            print(f"Error loading WikiText-103: {e}")
            # Fallback to dummy data for demonstration
            fallback_size = max_samples if max_samples is not None else 1000
            self.dataset = [{'text': f'This is sample text {i} for demonstration purposes.'}
                            for i in range(fallback_size)]

        print(f"Loaded {len(self.dataset)} text samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self.dataset[idx], 'keys'):
            text = self.dataset[idx]['text']
        else:
            text = self.dataset[idx]['text']

        # Tokenize
        tokens = self.tokenizer_wrapper.encode(text, self.max_length)

        # Apply perturbation if specified
        if self.perturbation_type and random.random() < self.perturbation_prob:
            tokens = self._apply_perturbation(tokens)

        # Create input and target for next-token prediction
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'labels': torch.tensor(target_tokens, dtype=torch.long),
            'original_text': text
        }

    def _apply_perturbation(self, tokens):
        vocab_size = self.tokenizer_wrapper.get_vocab_size()
        pad_token = 0  # Assuming pad token is 0

        if self.perturbation_type == 'remove_random':
            return TextPerturbations.remove_random_tokens(
                tokens, self.perturbation_params.get('p', 0.1), pad_token
            )
        elif self.perturbation_type == 'replace_random':
            return TextPerturbations.replace_random_tokens(
                tokens, self.perturbation_params.get('p', 0.1), vocab_size, pad_token
            )
        elif self.perturbation_type == 'block_shuffle':
            return TextPerturbations.block_shuffle(
                tokens, self.perturbation_params.get('k', 4), pad_token
            )
        elif self.perturbation_type == 'plug_random':
            return TextPerturbations.plug_random_tokens(
                tokens, self.perturbation_params.get('p', 0.1), vocab_size,
                self.max_length, pad_token
            )
        else:
            return tokens


class MNISTDataset(Dataset):
    """MNIST dataset with perturbations - now uses full dataset"""

    def __init__(self, root='./data', train=True, transform=None,
                 perturbation_type=None, perturbation_prob=0.1, perturbation_params=None):
        self.perturbation_type = perturbation_type
        self.perturbation_prob = perturbation_prob
        self.perturbation_params = perturbation_params or {}

        # Default transform
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Load full MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        print(f"Loaded {len(self.dataset)} MNIST samples ({'train' if train else 'test'} set)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        original_image = image.clone()

        # Apply perturbation if specified
        if self.perturbation_type and random.random() < self.perturbation_prob:
            image = self._apply_perturbation(image)

        return {
            'image': image,
            'original_image': original_image,
            'label': label
        }

    def _apply_perturbation(self, image):
        if self.perturbation_type == 'remove_random':
            return ImagePerturbations.remove_random_pixels(
                image, self.perturbation_params.get('p', 0.1)
            )
        elif self.perturbation_type == 'replace_random':
            return ImagePerturbations.replace_random_pixels(
                image, self.perturbation_params.get('p', 0.1)
            )
        elif self.perturbation_type == 'block_shuffle':
            return ImagePerturbations.block_shuffle(
                image, self.perturbation_params.get('k', 4)
            )
        elif self.perturbation_type == 'plug_random':
            return ImagePerturbations.plug_random_pixels(
                image, self.perturbation_params.get('p', 0.1)
            )
        else:
            return image


class DataPipeline:
    """Main pipeline class to handle both datasets"""

    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.dataloaders = {}

    def setup_wikitext(self, splits=['train', 'validation'], **kwargs):
        """Setup WikiText datasets with a limited number of samples"""
        for split in splits:
            print(f"Loading WikiText-103 {split} split")

            # Set max samples
            max_samples = 10000 if split == 'train' else 1000

            # Initialize WikiTextDataset with max_samples parameter
            wikitext_dataset = WikiTextDataset(
                split=split,
                max_length=self.config.get('max_length', 512),
                tokenizer_type=self.config.get('tokenizer_type', 'gpt2'),
                perturbation_type=self.config.get('perturbation_type'),
                perturbation_prob=self.config.get('perturbation_prob', 0.1),
                perturbation_params=self.config.get('perturbation_params', {}),
                max_samples=max_samples  # Add this parameter
            )

            # Create DataLoader with MPS-compatible settings
            dataloader = DataLoader(
                wikitext_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=(split == 'train'),
                num_workers=0,  # MPS compatibility
                pin_memory=(self.config.get('device', 'cpu') != 'cpu')
            )

            self.datasets[f'wikitext_{split}'] = wikitext_dataset
            self.dataloaders[f'wikitext_{split}'] = dataloader
            print(f"Loaded WikiText-103 {split} split with {len(wikitext_dataset)} samples")

    def setup_mnist(self, splits=['train', 'test'], **kwargs):
        """Setup MNIST datasets - now uses full dataset"""
        for split in splits:
            print(f"Loading MNIST {split} split")
            is_train = (split == 'train')

            # Load the full MNIST dataset (no sample limiting)
            mnist_dataset = MNISTDataset(
                root=self.config.get('data_root', './data'),
                train=is_train,
                perturbation_type=self.config.get('perturbation_type'),
                perturbation_prob=self.config.get('perturbation_prob', 0.1),
                perturbation_params=self.config.get('perturbation_params', {})
            )

            # Create DataLoader with MPS-compatible settings
            dataloader = DataLoader(
                mnist_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=is_train,
                num_workers=0,  # MPS compatibility
                pin_memory=(self.config.get('device', 'cpu') != 'cpu')
            )

            self.datasets[f'mnist_{split}'] = mnist_dataset
            self.dataloaders[f'mnist_{split}'] = dataloader
            print(f"Loaded MNIST {split} split with {len(mnist_dataset)} samples")

    def get_dataloader(self, dataset_name: str):
        """Get dataloader by name"""
        return self.dataloaders.get(dataset_name)

    def get_dataset(self, dataset_name: str):
        """Get dataset by name"""
        return self.datasets.get(dataset_name)


def demo_pipeline():
    """Demonstration of the pipeline"""

    # Configuration
    config = {
        'batch_size': 16,
        'max_length': 256,
        'tokenizer_type': 'gpt2',
        'num_workers': 0,  # Set to 0 for demo to avoid multiprocessing issues
        'data_root': './data'
    }

    print("=== Data Pipeline Demo ===\n")

    # Test different perturbation types
    perturbation_types = [
        ('remove_random', {'p': 0.15}),
        ('replace_random', {'p': 0.1}),
        ('block_shuffle', {'k': 4}),
        ('plug_random', {'p': 0.1}),
    ]

    for perturbation_type, params in perturbation_types:
        print(f"--- Testing {perturbation_type} ---")

        # Update config
        test_config = config.copy()
        test_config.update({
            'perturbation_type': perturbation_type,
            'perturbation_prob': 1.0,  # Always apply for demo
            'perturbation_params': params
        })

        # Initialize pipeline
        pipeline = DataPipeline(test_config)

        # Setup WikiText
        try:
            pipeline.setup_wikitext(['train'])
            wikitext_loader = pipeline.get_dataloader('wikitext_train')

            # Test one batch
            batch = next(iter(wikitext_loader))
            print(f"WikiText batch shape: {batch['input_ids'].shape}")
            print(f"Sample original text: {batch['original_text'][0][:100]}...")

            # Decode first sample to show perturbation effect
            tokenizer = pipeline.get_dataset('wikitext_train').tokenizer_wrapper
            original_tokens = torch.cat([batch['input_ids'][0], batch['labels'][0][-1:]])
            decoded_text = tokenizer.decode(original_tokens.tolist())
            print(f"Sample decoded text: {decoded_text[:100]}...")

        except Exception as e:
            print(f"WikiText test failed: {e}")

        # Setup MNIST - Now uses full dataset
        try:
            pipeline.setup_mnist(['train'])
            mnist_loader = pipeline.get_dataloader('mnist_train')

            # Test one batch
            batch = next(iter(mnist_loader))
            print(f"MNIST batch shape: {batch['image'].shape}")
            print(f"Labels: {batch['label'][:5].tolist()}")
            print(f"Full MNIST train dataset size: {len(pipeline.get_dataset('mnist_train'))}")

            # Check perturbation effect
            original_mean = batch['original_image'].mean().item()
            perturbed_mean = batch['image'].mean().item()
            print(f"Mean pixel change: {abs(original_mean - perturbed_mean):.4f}")

        except Exception as e:
            print(f"MNIST test failed: {e}")

        print()

