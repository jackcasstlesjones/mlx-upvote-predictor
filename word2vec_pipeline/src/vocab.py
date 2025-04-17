from collections import Counter
import pickle
import numpy as np


class Vocabulary:
    def __init__(self, min_freq=5):
        self.word2idx = {}
        self.idx2word = {}
        self.frequencies = {}
        self.min_freq = min_freq
        self.sampling_table = None

    def build(self, token_stream):
        """
        Build vocabulary from token stream

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

    def create_sampling_table(self, t=1e-5):
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

    def get_index(self, word):
        """Get index for word, return <UNK> index if not found"""
        return self.word2idx.get(word, 0)

    def get_word(self, idx):
        """Get word for index"""
        return self.idx2word.get(idx, "<UNK>")

    def convert_tokens_to_ids(self, tokens):
        """Convert a list of tokens to their corresponding indices"""
        return [self.get_index(token) for token in tokens]

    def subsample_tokens(self, token_ids):
        """
        Apply subsampling to tokens based on frequency

        Args:
            token_ids: List of token IDs

        Returns:
            list: Filtered list of token IDs
        """
        if not self.sampling_table:
            return token_ids

        return [
            idx for idx in token_ids
            if np.random.random() > self.sampling_table.get(idx, 0)
        ]

    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)

    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
