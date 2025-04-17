#!/usr/bin/env python
"""
Evaluation script for Word2Vec models.

This script evaluates a trained Word2Vec model against standard
and domain-specific benchmarks, generating a comprehensive report
that can be used to assess model quality before export.
"""

from src.eval_utils import evaluate_embeddings
import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Word2Vec embeddings")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with model files (embeddings.npy and vocab.json)"
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npy file (overrides model_dir)"
    )

    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to vocabulary .json file (overrides model_dir)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (.json)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Quality threshold (0-1) for passing evaluation"
    )

    return parser.parse_args()


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of evaluation results.

    Args:
        results: Evaluation results dictionary
    """
    print("\n=== Word2Vec Evaluation Summary ===\n")

    # Overall scores
    print("Overall Scores:")
    print(
        "  Standard benchmark score: "
        f"{results['overall_scores']['standard_score']:.4f}")
    print(
        "  Tech domain score: "
        f"{results['overall_scores']['tech_domain_score']:.4f}")
    print(
        f"  Combined score: {results['overall_scores']['combined_score']:.4f}")

    # Standard evaluation highlights
    std_eval = results['standard_evaluation']
    print("\nStandard Benchmarks:")
    print(
        "  Word similarity: "
        f"{std_eval['word_similarity']['average_similarity']:.4f}"
    )
    print(f"  Analogy accuracy: {std_eval['analogies']['accuracy']:.4f}")
    print(
        "  Cluster separation: "
        f"{std_eval['clustering']['cluster_separation']:.4f}"
    )
    print(
        "  Vocabulary coverage: "
        f"{std_eval['vocabulary_coverage']['standard_words_coverage']:.2%}"
    )

    # Tech domain highlights
    tech_eval = results['tech_domain_evaluation']
    print("\nTech Domain Benchmarks:")
    print(
        f"  Tech word similarity: "
        f"{tech_eval['tech_word_similarity']['average_similarity']:.4f}"
    )
    print(
        "  Tech analogy accuracy: "
        f"{tech_eval['tech_analogies']['accuracy']:.4f}"
    )
    print(
        "  Tech cluster separation: "
        f"{tech_eval['tech_clustering']['cluster_separation']:.4f}"
    )
    print(
        "  Tech vocabulary coverage: "
        f"{tech_eval['tech_vocabulary_coverage']['tech_domain_coverage']:.2%}"
    )

    # Unknown words summary
    std_unknown = (
        len(std_eval['word_similarity']['unknown_words']) +
        len(std_eval['analogies']['unknown_words']) +
        len(std_eval['clustering']['unknown_words']) +
        len(std_eval['nearest_neighbors']['unknown_words'])
    )

    tech_unknown = (
        len(tech_eval['tech_word_similarity']['unknown_words']) +
        len(tech_eval['tech_analogies']['unknown_words']) +
        len(tech_eval['tech_clustering']['unknown_words']) +
        len(tech_eval['tech_nearest_neighbors']['unknown_words'])
    )

    print("\nUnknown Words:")
    print(f"  Standard benchmarks: {std_unknown} words not in vocabulary")
    print(f"  Tech domain benchmarks: {tech_unknown} words not in vocabulary")

    # Sample nearest neighbors
    if 'computer' in std_eval['nearest_neighbors']['neighbors']:
        neighbors = std_eval['nearest_neighbors']['neighbors']['computer']
        print("\nSample nearest neighbors for 'computer':")
        for i, neighbor in enumerate(neighbors[:5]):
            print(
                f"  {i+1}. {neighbor['word']} ({neighbor['similarity']:.4f})")

    print("\n===================================\n")


def main() -> None:
    """Main entry point."""
    # Setup
    setup_logging()
    args = parse_args()

    # Determine file paths
    embeddings_path = args.embeddings
    vocab_path = args.vocab

    if not embeddings_path:
        embeddings_path = os.path.join(args.model_dir, "embeddings.npy")

    if not vocab_path:
        vocab_path = os.path.join(args.model_dir, "vocab.json")

    # Set default output path if not provided
    output_path = args.output
    if not output_path:
        output_dir = os.path.join(args.model_dir, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluation_results.json")

    # Check if files exist
    if not os.path.exists(embeddings_path):
        logging.error(f"Embeddings file not found: {embeddings_path}")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        logging.error(f"Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    # Run evaluation
    logging.info(f"Evaluating embeddings from {embeddings_path}")
    logging.info(f"Using vocabulary from {vocab_path}")

    results = evaluate_embeddings(
        embeddings_path=embeddings_path,
        vocab_path=vocab_path,
        output_path=output_path
    )

    # Print summary
    print_summary(results)

    # Check against threshold
    combined_score = results['overall_scores']['combined_score']
    threshold = args.threshold

    if combined_score >= threshold:
        logging.info(
            f"Model passed evaluation: {combined_score:.4f} >= {threshold:.4f}"
        )
        print(
            f"\n✅ Model PASSED evaluation (score: {combined_score:.4f}, "
            f"threshold: {threshold:.4f})"
        )
        sys.exit(0)
    else:
        logging.warning(
            f"Model failed evaluation: {combined_score:.4f} < {threshold:.4f}")
        print(
            f"\n❌ Model FAILED evaluation (score: {combined_score:.4f}, "
            f"threshold: {threshold:.4f})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
