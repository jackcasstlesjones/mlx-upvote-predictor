#!/usr/bin/env python
"""
Test word embeddings for semantic similarity
"""

import os
import pickle
import sys

import numpy as np

# Add the parent directory to Python path to make src module available

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import our vocabulary class


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    # Paths
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "embeddings", "word_vectors.npy")
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "processed", "vocab.pkl")

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Define test word pairs
    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("city", "town"),
        ("big", "large"),
        ("small", "tiny"),
        ("machine", "learning"),
        ("essay", "dissertation"),
        ("small", "tiny"),
        ("keyboard", "mouse"),

    ]

    # Test similarity between word pairs
    print("\nWord pair similarities:")
    similarities = []

    for word1, word2 in word_pairs:
        idx1 = vocab.get_index(word1)
        idx2 = vocab.get_index(word2)

        if idx1 == 0:  # <UNK> token
            print(f"Word '{word1}' not found in vocabulary")
            continue

        if idx2 == 0:  # <UNK> token
            print(f"Word '{word2}' not found in vocabulary")
            continue

        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]

        similarity = cosine_similarity(vec1, vec2)
        similarities.append(similarity)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

    if similarities:
        avg_similarity = np.mean(similarities)
        print(f"\nAverage similarity: {avg_similarity:.4f}")

    # Find similar words for some examples
    test_words = ["water", "king", "computer", "day", "problem", "octopus"]

    for query_word in test_words:
        idx = vocab.get_index(query_word)

        if idx == 0:  # <UNK> token
            print(f"\nWord '{query_word}' not found in vocabulary")
            continue

        print(f"\nTop 10 words similar to '{query_word}':")
        query_vec = embeddings[idx]

        # Compute similarity to all words
        word_similarities = []

        for word, word_idx in vocab.word2idx.items():
            if word == query_word or word == "<UNK>":
                continue

            word_vec = embeddings[word_idx]
            similarity = cosine_similarity(query_vec, word_vec)
            word_similarities.append((word, similarity))

        # Sort by similarity and print top 10
        word_similarities.sort(key=lambda x: x[1], reverse=True)

        for word, similarity in word_similarities[:10]:
            print(f"{word}: {similarity:.4f}")


if __name__ == "__main__":
    main()
