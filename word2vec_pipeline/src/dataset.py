import random
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Generator, List, Tuple, Callable, Iterator, Any


def generate_skipgram_pairs(
        tokens: List[str],
        vocab: Any,
        window_size: int = 5
) -> Generator[Tuple[int, int], None, None]:
    """
    Generate skip-gram pairs from a list of tokens.

    For each token, creates (center, context) pairs with other tokens
    in a dynamic window.

    Args:
        tokens: List of tokens
        vocab: Vocabulary object with token conversion methods
        window_size: Maximum context window size

    Yields:
        tuple: (center_word_id, context_word_id) pairs
    """
    # Convert tokens to IDs and apply subsampling
    token_ids = vocab.convert_tokens_to_ids(tokens)
    token_ids = vocab.subsample_tokens(token_ids)

    for i, center in enumerate(token_ids):
        # Randomly select window size for current center word
        window = random.randint(1, window_size)

        # Generate pairs for all context words within window
        for j in range(
            max(0, i - window),
            min(len(token_ids), i + window + 1)
        ):
            if i != j:  # Skip the center word itself
                context = token_ids[j]
                yield (center, context)


class SkipgramDataset(IterableDataset):
    """
    PyTorch IterableDataset for Skip-gram model training.

    Generates training samples from token streams with negative sampling,
    optimized for memory efficiency.
    """

    def __init__(self, token_stream: Callable[[], Iterator[List[str]]],
                 vocab: Any, window_size: int = 5, neg_samples: int = 5):
        """
        Initialize the Skip-gram dataset.

        Args:
            token_stream: Generator yielding lists of tokens
            vocab: Vocabulary object with token conversion methods
            window_size: Maximum context window size
            neg_samples: Number of negative samples per positive sample
        """
        self.token_stream = token_stream
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.vocab_size = len(vocab)

        # Create negative sampling distribution (unigram^0.75)
        self._create_negative_sampling_table()

    def _create_negative_sampling_table(
            self,
            table_size: int = 100000000
    ) -> None:
        """
        Create table for negative sampling using frequency distribution.

        Implements the unigram distribution raised to power of 0.75
        as described in the original Word2Vec paper.

        Args:
            table_size: Size of the sampling table
        """
        vocab_size = len(self.vocab)
        sampling_weights = np.zeros(vocab_size)

        # Get word frequencies for all words in vocabulary
        for idx in range(vocab_size):
            freq = self.vocab.frequencies.get(idx, 0)
            sampling_weights[idx] = freq ** 0.75

        # Normalize
        sampling_weights = sampling_weights / np.sum(sampling_weights)

        # Create sampling table
        self.neg_sampling_table = np.random.choice(
            np.arange(vocab_size),
            size=table_size,
            p=sampling_weights,
            replace=True
        )

    def _get_negative_samples(
            self,
            positive_idx: int,
            n_samples: int
    ) -> np.ndarray:
        """
        Get negative samples from precomputed table, avoiding the
        positive sample.

        Args:
            positive_idx: Index to exclude from negative samples
            n_samples: Number of negative samples to return

        Returns:
            np.ndarray: Array of negative sample indices
        """
        indices = np.random.randint(
            0, len(self.neg_sampling_table), size=n_samples + 10)
        samples = self.neg_sampling_table[indices]

        # Filter out the positive sample
        samples = samples[samples != positive_idx]

        # If we filtered too many, get more samples
        if len(samples) < n_samples:
            more_samples = self._get_negative_samples(
                positive_idx, n_samples - len(samples))
            samples = np.concatenate([samples, more_samples])

        return samples[:n_samples]

    def __iter__(self) -> Iterator[Tuple[int, int, np.ndarray]]:
        """
        Iterate through token stream and yield batches.

        Yields:
            tuple: (center, context, negatives) triplets where:
                - center is the index of the center word
                - context is the index of the context word
                - negatives is an array of negative sample indices
        """
        for tokens in self.token_stream():
            for center, context in generate_skipgram_pairs(
                tokens,
                self.vocab,
                self.window_size
            ):
                neg_samples = self._get_negative_samples(
                    context, self.neg_samples)
                yield center, context, neg_samples


def create_dataloader(dataset: SkipgramDataset, batch_size: int = 512,
                      num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the SkipgramDataset.

    Args:
        dataset: SkipgramDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    # Create a batch collation function
    def collate_fn(
            batch: List[Tuple[int, int, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers, contexts, negatives = zip(*batch)
        return (
            torch.LongTensor(centers),
            torch.LongTensor(contexts),
            torch.LongTensor(negatives)
        )

    # We need to use IterableDataset approach
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,  # Cannot shuffle an IterableDataset
    )
