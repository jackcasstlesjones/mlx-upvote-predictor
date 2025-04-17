import random
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def generate_skipgram_pairs(tokens, vocab, window_size=5):
    """
    Generate skip-gram pairs from a list of tokens

    Args:
        tokens: List of tokens
        vocab: Vocabulary object
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
    def __init__(self, token_stream, vocab, window_size=5, neg_samples=5):
        """
        Dataset for Skip-gram model training

        Args:
            token_stream: Generator yielding lists of tokens
            vocab: Vocabulary object
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

    def _create_negative_sampling_table(self, table_size=100000000):
        """
        Create table for negative sampling, using unigram
        distribution raised to power of 0.75
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

    def _get_negative_samples(self, positive_idx, n_samples):
        """
        Get negative samples from precomputed table,
        avoiding the positive sample
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

    def __iter__(self):
        """
        Iterate through token stream and yield
        batches of (center, context, negatives)
        """
        for tokens in self.token_stream:
            for center, context in generate_skipgram_pairs(
                tokens,
                self.vocab,
                self.window_size
            ):
                neg_samples = self._get_negative_samples(
                    context, self.neg_samples)
                yield center, context, neg_samples


def create_dataloader(dataset, batch_size=512, num_workers=4):
    """
    Create a DataLoader for the SkipgramDataset

    Args:
        dataset: SkipgramDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """

    # Create a batch collation function
    def collate_fn(batch):
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
