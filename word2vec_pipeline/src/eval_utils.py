"""
Evaluation utilities for Word2Vec embeddings.

This module provides functions to evaluate word embeddings quality
through various methods including:
- Word similarity tasks
- Analogy resolution
- Domain-specific term clustering
- Nearest neighbor evaluation
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Word2VecEvaluator:
    """
    Evaluator for Word2Vec embeddings quality.

    Provides methods to assess embedding quality using standard
    benchmarks and domain-specific metrics.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        unknown_token: str = "<UNK>"
    ):
        """
        Initialize evaluator with embeddings and vocabulary mappings.

        Args:
            embeddings: Word embedding matrix of shape (vocab_size, embed_dim)
            word2idx: Dictionary mapping words to indices
            idx2word: Dictionary mapping indices to words
            unknown_token: Token used for unknown words
        """
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.unknown_token = unknown_token
        self.logger = logging.getLogger(__name__)

        # Cache for word vectors (to avoid repeated lookups)
        self.word_vector_cache = {}
        self.unknown_idx = self.word2idx.get(unknown_token, 0)

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get embedding vector for a word, with caching.

        Args:
            word: Input word

        Returns:
            np.ndarray: Word embedding vector
        """
        if word in self.word_vector_cache:
            return self.word_vector_cache[word]

        idx = self.word2idx.get(word, self.unknown_idx)
        vector = self.embeddings[idx].copy()
        self.word_vector_cache[word] = vector
        return vector

    def cosine_sim(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def evaluate_word_pairs(
            self,
            pairs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate similarity between word pairs.

        Args:
            pairs: List of (word1, word2) tuples to evaluate

        Returns:
            Dict: Results including average similarity and per-pair scores
        """
        similarities = []
        pair_results = []
        unknown_words = set()

        for word1, word2 in pairs:
            # Check if words are in vocabulary
            if word1 not in self.word2idx:
                unknown_words.add(word1)
            if word2 not in self.word2idx:
                unknown_words.add(word2)

            # Calculate similarity
            sim = self.cosine_sim(word1, word2)
            similarities.append(sim)
            pair_results.append({
                "word1": word1,
                "word2": word2,
                "similarity": sim
            })

        # Calculate statistics
        avg_sim = np.mean(similarities) if similarities else 0.0
        std_sim = np.std(similarities) if similarities else 0.0

        return {
            "average_similarity": avg_sim,
            "std_similarity": std_sim,
            "unknown_words": list(unknown_words),
            "unknown_ratio": len(unknown_words) / (2 * len(pairs))
            if pairs else 0,
            "pair_results": pair_results
        }

    def evaluate_analogies(
            self,
            analogies: List[Tuple[str, str, str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate analogy task (a is to b as c is to ?)

        Args:
            analogies: List of (a, b, c, d) analogy tuples

        Returns:
            Dict: Results including accuracy and per-analogy scores
        """
        correct = 0
        results = []
        unknown_words = set()

        for a, b, c, d in analogies:
            # Check for unknown words
            for word in (a, b, c, d):
                if word not in self.word2idx:
                    unknown_words.add(word)

            # Get vectors
            a_vec = self.get_word_vector(a)
            b_vec = self.get_word_vector(b)
            c_vec = self.get_word_vector(c)

            # Skip if any word is unknown
            if (a not in self.word2idx or
                b not in self.word2idx or
                    c not in self.word2idx):
                results.append({
                    "analogy": (a, b, c, d),
                    "predicted": None,
                    "correct": False,
                    "contains_unknown": True
                })
                continue

            # Calculate target vector: b - a + c
            target_vec = b_vec - a_vec + c_vec

            # Find nearest word to target, excluding input words
            exclude = {self.word2idx.get(w, -1) for w in [a, b, c]}

            # Compute similarity to all words
            sims = cosine_similarity(
                target_vec.reshape(1, -1),
                self.embeddings
            )[0]

            # Set similarity of input words to -inf to exclude them
            for idx in exclude:
                if idx >= 0 and idx < len(sims):
                    sims[idx] = -float('inf')

            # Get most similar word
            pred_idx = np.argmax(sims)
            pred_word = self.idx2word.get(pred_idx, self.unknown_token)

            # Check if prediction is correct
            is_correct = pred_word == d
            if is_correct:
                correct += 1

            # Store result
            results.append({
                "analogy": (a, b, c, d),
                "predicted": pred_word,
                "correct": is_correct,
                "contains_unknown": False
            })

        # Calculate accuracy
        valid_count = len(analogies) - \
            sum(1 for r in results if r["contains_unknown"])
        accuracy = correct / valid_count if valid_count > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_valid": valid_count,
            "unknown_words": list(unknown_words),
            "unknown_ratio": len(unknown_words) / (4 * len(analogies))
            if analogies else 0,
            "results": results
        }

    def evaluate_clustering(
            self,
            word_groups: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Evaluate how well semantically related words cluster together.

        Args:
            word_groups: Dictionary mapping categories to lists of words

        Returns:
            Dict: Results with intra-group and inter-group similarities
        """
        unknown_words = set()
        group_vectors = {}

        # Calculate average vector for each group
        for group_name, words in word_groups.items():
            # Track unknown words
            group_unknown = [w for w in words if w not in self.word2idx]
            unknown_words.update(group_unknown)

            # Get vectors for known words
            known_words = [w for w in words if w in self.word2idx]
            if not known_words:
                continue

            word_vectors = np.stack([self.get_word_vector(w)
                                    for w in known_words])
            group_vector = np.mean(word_vectors, axis=0)
            group_vectors[group_name] = group_vector

        # Calculate intra-group similarity
        # (how similar words are within groups)
        intra_group_sims = {}
        for group_name, words in word_groups.items():
            if group_name not in group_vectors:
                continue

            # Get known words in group
            known_words = [w for w in words if w in self.word2idx]
            if len(known_words) < 2:
                continue

            # Calculate pairwise similarities within group
            pairs = [(w1, w2) for i, w1 in enumerate(known_words)
                     for w2 in known_words[i+1:]]
            pair_sims = [self.cosine_sim(w1, w2) for w1, w2 in pairs]

            intra_group_sims[group_name] = {
                "mean": np.mean(pair_sims),
                "min": np.min(pair_sims),
                "max": np.max(pair_sims)
            }

        # Calculate inter-group similarity
        # (how distinct groups are from each other)
        inter_group_sims = {}
        group_names = list(group_vectors.keys())
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i+1:]:
                vec1 = group_vectors[group1]
                vec2 = group_vectors[group2]

                # Calculate cosine similarity between group vectors
                sim = cosine_similarity(
                    vec1.reshape(1, -1),
                    vec2.reshape(1, -1)
                )[0][0]

                key = f"{group1}_vs_{group2}"
                inter_group_sims[key] = float(sim)

        # Calculate overall metrics
        intra_mean = np.mean(
            [g["mean"] for g in intra_group_sims.values()]
        ) if intra_group_sims else 0
        inter_mean = np.mean(list(inter_group_sims.values())
                             ) if inter_group_sims else 0

        # Higher separation indicates better clustering
        separation = intra_mean - inter_mean

        return {
            "cluster_separation": separation,
            "intra_group_similarity": intra_mean,
            "inter_group_similarity": inter_mean,
            "intra_group_details": intra_group_sims,
            "inter_group_details": inter_group_sims,
            "unknown_words": list(unknown_words),
            "unknown_ratio":
            len(unknown_words) /
            sum(len(words) for words in word_groups.values())
        }

    def nearest_neighbors(
            self,
            word: str,
            n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find nearest neighbors for a word.

        Args:
            word: Query word
            n: Number of neighbors to return

        Returns:
            List[Dict]: Nearest neighbors with similarity scores
        """
        if word not in self.word2idx:
            return []

        word_idx = self.word2idx[word]
        word_vec = self.embeddings[word_idx]

        # Calculate similarities to all words
        sims = cosine_similarity(
            word_vec.reshape(1, -1),
            self.embeddings
        )[0]

        # Get indices of top n+1 similar words (including the word itself)
        top_indices = np.argsort(sims)[::-1][:n+1]

        # Remove the word itself if present
        neighbors = [
            {
                "word": self.idx2word[idx],
                "similarity": float(sims[idx])
            }
            for idx in top_indices
            if idx != word_idx and idx in self.idx2word
        ]

        return neighbors[:n]  # Limit to n neighbors

    def evaluate_nearest_neighbors(
            self,
            words: List[str],
            n: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate nearest neighbors for a list of words.

        Args:
            words: List of query words
            n: Number of neighbors to find for each word

        Returns:
            Dict: Results including neighbors for each word
        """
        results = {}
        unknown_words = []

        for word in words:
            if word not in self.word2idx:
                unknown_words.append(word)
                results[word] = []
                continue

            results[word] = self.nearest_neighbors(word, n)

        return {
            "neighbors": results,
            "unknown_words": unknown_words,
            "unknown_ratio": len(unknown_words) / len(words) if words else 0
        }

    def run_standard_evaluation(self) -> Dict[str, Any]:
        """
        Run standard evaluation suite.

        Returns:
            Dict: Comprehensive evaluation results
        """
        # Define standard word pairs
        word_pairs = [
            ("king", "queen"), ("man", "woman"),
            ("good", "bad"), ("buy", "sell"),
            ("fast", "slow"), ("rich", "poor"),
            ("cat", "dog"), ("happy", "sad"),
            ("hot", "cold"), ("big", "small")
        ]

        # Define standard analogies
        analogies = [
            ("man", "woman", "king", "queen"),
            ("city", "cities", "child", "children"),
            ("good", "better", "bad", "worse"),
            ("buy", "bought", "sell", "sold"),
            ("france", "paris", "italy", "rome"),
            ("small", "smaller", "big", "bigger")
        ]

        # Define word groups for clustering
        word_groups = {
            "animals": [
                "cat",
                "dog",
                "horse",
                "cow",
                "sheep",
                "pig",
                "lion",
                "tiger"
            ],
            "technology": [
                "computer",
                "software",
                "hardware",
                "internet",
                "code",
                "program"
            ],
            "food": [
                "bread",
                "butter",
                "cheese",
                "meat",
                "vegetable",
                "fruit"
            ],
            "colors": ["red", "blue", "green", "yellow", "black", "white"]
        }

        # Words for nearest neighbor evaluation
        nn_words = ["computer", "king", "money", "science", "book"]

        # Run evaluations
        pair_results = self.evaluate_word_pairs(word_pairs)
        analogy_results = self.evaluate_analogies(analogies)
        cluster_results = self.evaluate_clustering(word_groups)
        nn_results = self.evaluate_nearest_neighbors(nn_words)

        # Combine results
        results = {
            "word_similarity": pair_results,
            "analogies": analogy_results,
            "clustering": cluster_results,
            "nearest_neighbors": nn_results,
            "vocabulary_coverage": {
                "total_vocab_size": len(self.word2idx),
                "standard_words_coverage": 1.0 - (
                    pair_results["unknown_ratio"] +
                    analogy_results["unknown_ratio"] +
                    cluster_results["unknown_ratio"] +
                    nn_results["unknown_ratio"]
                ) / 4
            }
        }

        return results

    def run_tech_domain_evaluation(self) -> Dict[str, Any]:
        """
        Run technology domain-specific evaluation.

        Returns:
            Dict: Domain-specific evaluation results
        """
        # Tech domain word pairs
        tech_pairs = [
            ("python", "programming"), ("code", "algorithm"),
            ("data", "database"), ("cloud", "server"),
            ("network", "internet"), ("software", "application"),
            ("mobile", "phone"), ("security", "encryption"),
            ("neural", "network"), ("machine", "learning")
        ]

        # Tech domain analogies
        tech_analogies = [
            ("python", "django", "ruby", "rails"),
            ("java", "object", "javascript", "prototype"),
            ("data", "database", "file", "filesystem"),
            ("code", "software", "circuit", "hardware"),
            ("google", "search", "microsoft", "windows")
        ]

        # Tech domain word groups
        tech_groups = {
            "programming_languages": [
                "python", "javascript", "java", "cpp", "ruby", "golang", "rust"
            ],
            "web_technologies": [
                "html", "css", "api", "rest", "http", "browser", "dom"
            ],
            "data_science": [
                "data",
                "algorithm",
                "neural",
                "statistical",
                "regression",
                "training"
            ],
            "hardware": [
                "cpu", "gpu", "memory", "ram", "storage", "server", "cloud"
            ]
        }

        # Tech domain test words for nearest neighbors
        tech_nn_words = ["algorithm", "python", "security", "data", "cloud"]

        # Run domain evaluations
        pair_results = self.evaluate_word_pairs(tech_pairs)
        analogy_results = self.evaluate_analogies(tech_analogies)
        cluster_results = self.evaluate_clustering(tech_groups)
        nn_results = self.evaluate_nearest_neighbors(tech_nn_words)

        # Combine results
        results = {
            "tech_word_similarity": pair_results,
            "tech_analogies": analogy_results,
            "tech_clustering": cluster_results,
            "tech_nearest_neighbors": nn_results,
            "tech_vocabulary_coverage": {
                "total_vocab_size": len(self.word2idx),
                "tech_domain_coverage": 1.0 - (
                    pair_results["unknown_ratio"] +
                    analogy_results["unknown_ratio"] +
                    cluster_results["unknown_ratio"] +
                    nn_results["unknown_ratio"]
                ) / 4
            }
        }

        return results

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including standard
        and domain-specific tests.

        Returns:
            Dict: Complete evaluation results
        """
        standard_results = self.run_standard_evaluation()
        tech_results = self.run_tech_domain_evaluation()

        # Calculate overall scores
        standard_score = (
            standard_results["word_similarity"]["average_similarity"] +
            standard_results["analogies"]["accuracy"] +
            max(0, standard_results["clustering"]["cluster_separation"]) +
            standard_results["vocabulary_coverage"]["standard_words_coverage"]
        ) / 4

        tech_score = (
            tech_results["tech_word_similarity"]["average_similarity"] +
            tech_results["tech_analogies"]["accuracy"] +
            max(0, tech_results["tech_clustering"]["cluster_separation"]) +
            tech_results["tech_vocabulary_coverage"]["tech_domain_coverage"]
        ) / 4

        # Combine results
        results = {
            "standard_evaluation": standard_results,
            "tech_domain_evaluation": tech_results,
            "overall_scores": {
                "standard_score": standard_score,
                "tech_domain_score": tech_score,
                "combined_score": (standard_score + tech_score) / 2
            }
        }

        return results


def evaluate_embeddings(
    embeddings_path: str,
    vocab_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate embeddings from files and optionally save results.

    Args:
        embeddings_path: Path to embeddings .npy file
        vocab_path: Path to vocabulary .json file
        output_path: Path to save evaluation results (optional)

    Returns:
        Dict: Evaluation results
    """
    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    word2idx = vocab_data["word2idx"]
    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

    # Create evaluator
    evaluator = Word2VecEvaluator(embeddings, word2idx, idx2word)

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()

    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Saved evaluation results to {output_path}")

    return results
