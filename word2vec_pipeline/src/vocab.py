from collections import Counter
import pickle
import os
from typing import Dict, Generator, List, Optional
import numpy as np


class Vocabulary:
    """
    Vocabulary class for word2vec model.

    Handles token-to-index mapping, frequency counting, and subsampling
    for efficient training. Includes methods to save/load vocabulary
    and convert between tokens and indices.
    """

    def __init__(self, min_freq: int = 5):
        """
        Initialize vocabulary object.

        Args:
            min_freq: Minimum frequency threshold for words to be included
        """
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.frequencies: Dict[int, int] = {}
        self.min_freq: int = min_freq
        self.sampling_table: Optional[Dict[int, float]] = None

    def build(
        self,
        token_stream: Generator[List[str], None, None]
    ) -> 'Vocabulary':
        """
        Build vocabulary from token stream.

        Args:
            token_stream: Generator yielding lists of tokens

        Returns:
            self: Updated vocabulary object
        """
        counter = Counter()
        print("Counting tokens...")
        for tokens in token_stream:
            counter.update(tokens)

        print(f"Found {len(counter)} unique tokens")

        # Filter by minimum frequency and create word mappings
        filtered_words = [word for word,
                          count in counter.items() if count >= self.min_freq]
        print(
            f"Keeping {len(filtered_words)} tokens with "
            f"min frequency {self.min_freq}")

        # Add <UNK> token at index 0
        self.word2idx = {"<UNK>": 0}
        self.idx2word = {0: "<UNK>"}

        # Add remaining words
        for i, word in enumerate(filtered_words):
            self.word2idx[word] = i + 1
            self.idx2word[i + 1] = word

        # Store frequencies including <UNK>
        self.frequencies = {
            0: sum(
                count for word,
                count in counter.items() if count < self.min_freq
            )
        }
        for word, idx in self.word2idx.items():
            if word != "<UNK>":
                self.frequencies[idx] = counter[word]

        # Create subsampling probabilities
        self.create_sampling_table()

        return self

    def create_sampling_table(self, t: float = 1e-5) -> None:
        """
        Create subsampling table following word2vec paper:
        P(w) = 1 - sqrt(t / f(w))

        Args:
            t: Threshold parameter (default: 1e-5)
        """
        total_words = float(sum(self.frequencies.values()))
        self.sampling_table = {}

        for idx, count in self.frequencies.items():
            freq = count / total_words
            self.sampling_table[idx] = max(0, 1 - np.sqrt(t / freq))

    def get_index(self, word: str) -> int:
        """
        Get index for word, return <UNK> index if not found.

        Args:
            word: Input word

        Returns:
            int: Index of word or 0 for unknown words
        """
        return self.word2idx.get(word, 0)

    def get_word(self, idx: int) -> str:
        """
        Get word for index.

        Args:
            idx: Word index

        Returns:
            str: Word at given index or '<UNK>' if not found
        """
        return self.idx2word.get(idx, "<UNK>")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to their corresponding indices.

        Args:
            tokens: List of token strings

        Returns:
            List[int]: List of token indices
        """
        return [self.get_index(token) for token in tokens]

    def subsample_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Apply subsampling to tokens based on frequency.

        Implements the subsampling technique from the word2vec paper
        which helps balance the influence of common words.

        Args:
            token_ids: List of token IDs

        Returns:
            List[int]: Filtered list of token IDs
        """
        if not self.sampling_table:
            return token_ids

        return [
            idx for idx in token_ids
            if np.random.random() > self.sampling_table.get(idx, 0)
        ]

    def __len__(self) -> int:
        """
        Return vocabulary size.

        Returns:
            int: Number of tokens in vocabulary
        """
        return len(self.word2idx)

    def save(self, path: str) -> None:
        """
        Save vocabulary to file.

        Args:
            path: File path for saving the vocabulary
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """
        Load vocabulary from file.

        Args:
            path: File path to load vocabulary from

        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
