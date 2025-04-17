#!/usr/bin/env python
"""
Train Word2Vec model on the text8 dataset from Hugging Face.

This script handles the full Word2Vec training pipeline:
1. Data downloading and preprocessing
2. Vocabulary building
3. Model training
4. Embedding evaluation and export
"""

import os
import argparse
import yaml
import pickle
import logging
import json
from typing import Dict, Any, Optional, List

from datasets import load_dataset

from src.tokenize import load_and_tokenize_text8
from src.vocab import Vocabulary
from src.train import train_model
from src.utils import Timer
from src.eval_utils import evaluate_embeddings


def setup_logging() -> None:
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on text8 dataset")

    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model for production after training"
    )

    parser.add_argument(
        "--version",
        type=str,
        default="0.1.0",
        help="Model version for export"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set default values for missing parameters
    defaults = {
        "embed_dim": 100,
        "window_size": 5,
        "min_count": 5,
        "neg_samples": 10,
        "learning_rate": 0.002,
        "batch_size": 512,
        "epochs": 5,
        "checkpoint_dir": "checkpoints",
        "embeddings_dir": "embeddings",
        "export_dir": "exported_model",
        "vocab_path": "data/processed/vocab.pkl",
        "seed": 42
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            logging.info(f"Using default value for {key}: {value}")

    return config


def save_to_file(tokens: List[str], output_path: str) -> None:
    """
    Save tokens to a file, one sentence per line.

    Args:
        tokens: List of tokens
        output_path: Path to output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert the tokens to a single line of space-separated tokens
        f.write(" ".join(tokens))
    logging.info(f"Saved tokens to {output_path}")


def prepare_text8_data() -> str:
    """
    Download and prepare the text8 dataset.

    Returns:
        str: Path to processed tokens file
    """
    # Define paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    token_path = os.path.join(processed_dir, "text8_tokens.txt")

    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Check if processed file already exists
    if os.path.exists(token_path):
        logging.info(f"Using existing tokenized data at {token_path}")
        return token_path

    # Download and process the dataset
    logging.info("Downloading text8 dataset from Hugging Face...")
    dataset = load_dataset("afmck/text8")

    # Tokenize the data
    logging.info("Tokenizing data...")
    tokens = load_and_tokenize_text8(dataset)

    # Save tokens to file
    save_to_file(tokens, token_path)

    return token_path


def build_or_load_vocab(
        token_path: str,
        vocab_path: str,
        min_freq: int
) -> Vocabulary:
    """
    Build vocabulary from tokens or load from file if it exists.

    Args:
        token_path: Path to tokenized data
        vocab_path: Path to save/load vocabulary
        min_freq: Minimum word frequency

    Returns:
        Vocabulary: Vocabulary object
    """
    # Check if vocabulary exists
    if os.path.exists(vocab_path):
        logging.info(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    # Create vocabulary
    logging.info(f"Building vocabulary with min_freq={min_freq}...")

    # Define a generator function for the token stream
    def token_stream():
        with open(token_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip().split()

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(token_stream())

    # Save vocabulary
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    logging.info(f"Vocabulary built with {len(vocab)} words")
    logging.info(f"Saved vocabulary to {vocab_path}")

    return vocab


def prepare_export(
    model: Any,
    vocab: Vocabulary,
    config: Dict[str, Any],
    export_dir: str,
    version: str,
    eval_results: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Prepare model for export to production.

    Args:
        model: Trained model
        vocab: Vocabulary object
        config: Training configuration
        export_dir: Directory to export model
        version: Model version string
        eval_results: Evaluation results (optional)

    Returns:
        Dict[str, str]: Dictionary with paths to exported files
    """
    from src.utils import export_model

    logging.info(f"Exporting model to {export_dir}...")

    # Create export directory
    os.makedirs(export_dir, exist_ok=True)

    # Save model files
    paths = export_model(
        model,
        vocab,
        config,
        export_dir,
        version
    )

    # Add evaluation results if available
    if eval_results:
        eval_path = os.path.join(export_dir, "evaluation.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        paths["evaluation"] = eval_path
        logging.info(f"Saved evaluation results to {eval_path}")

    return paths


def main() -> None:
    """Main entry point for training script."""
    # Setup
    setup_logging()
    args = parse_args()
    timer = Timer()
    timer.start()

    # Load configuration
    config = load_config(args.config)

    # Prepare data
    token_path = prepare_text8_data()

    # Build or load vocabulary
    vocab_path = config.get("vocab_path", "data/processed/vocab.pkl")
    vocab = build_or_load_vocab(token_path, vocab_path, config["min_count"])

    # Define token stream function for training
    def get_token_stream():
        with open(token_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip().split()

    # Configure export directory if needed
    export_dir = None
    if args.export:
        export_dir = config.get("export_dir", "exported_model")

    # Train model
    logging.info("Starting model training...")
    model, training_info = train_model(
        token_stream_fn=get_token_stream,
        vocab=vocab,
        embed_dim=config["embed_dim"],
        window_size=config["window_size"],
        neg_samples=config["neg_samples"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        checkpoint_dir=config["checkpoint_dir"],
        embeddings_dir=config["embeddings_dir"],
        export_dir=export_dir,
        resume_from=args.resume,
        seed=config["seed"]
    )

    # Calculate training time
    train_time = timer.elapsed()
    logging.info(f"Training completed in {train_time:.2f}s")

    # Convert vocabulary for evaluation
    vocab_json_path = os.path.join(config["embeddings_dir"], "vocab.json")
    word2idx = vocab.word2idx
    idx2word = {str(k): v for k, v in vocab.idx2word.items()}

    with open(vocab_json_path, 'w') as f:
        json.dump({"word2idx": word2idx, "idx2word": idx2word}, f)

    logging.info(f"Saved vocabulary as JSON to {vocab_json_path}")

    # Run evaluation if not skipped
    eval_results = None
    if not args.skip_eval:
        logging.info("Running model evaluation...")
        eval_timer = Timer()
        eval_timer.start()

        # Get path to best embeddings
        embeddings_path = os.path.join(
            config["embeddings_dir"], "word_vectors.npy")

        # Run evaluation
        eval_results = evaluate_embeddings(
            embeddings_path=embeddings_path,
            vocab_path=vocab_json_path,
            output_path=os.path.join(
                config["embeddings_dir"], "evaluation.json")
        )

        eval_time = eval_timer.elapsed()
        logging.info(f"Evaluation completed in {eval_time:.2f}s")

        # Log summary
        combined_score = eval_results["overall_scores"]["combined_score"]
        logging.info(f"Model evaluation score: {combined_score:.4f}")

    # Export model if requested
    if args.export and not export_dir:
        export_dir = config.get("export_dir", "exported_model")
        export_timer = Timer()
        export_timer.start()

        # Export files
        export_paths = prepare_export(
            model,
            vocab,
            config,
            export_dir,
            args.version,
            eval_results
        )

        export_time = export_timer.elapsed()
        logging.info(f"Export completed in {export_time:.2f}s")
        logging.info(f"Model exported to {export_dir}")

    # Print final report
    total_time = timer.elapsed()

    print("\n=== Training Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Training time: {train_time:.2f}s")

    if not args.skip_eval:
        print(f"Evaluation time: {eval_time:.2f}s")
        print(
            "Evaluation score: "
            f"{eval_results['overall_scores']['combined_score']:.4f}"
        )

    if args.export:
        print(f"Export time: {export_time:.2f}s")
        print(f"Model exported to: {export_dir}")

    print("\nFinal model paths:")
    print(
        f"- Embeddings: "
        f"{os.path.join(config['embeddings_dir'], 'word_vectors.npy')}")
    print(f"- Vocabulary: {vocab_path}")
    print(f"- Best checkpoint: {training_info.get('best_model_path', 'None')}")

    if args.export:
        print("\nExported files:")
        for key, path in export_paths.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
