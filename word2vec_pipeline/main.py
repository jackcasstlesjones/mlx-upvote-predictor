#!/usr/bin/env python
"""
Word2Vec Pipeline: Training, Evaluation, and Export.

This script provides a unified workflow for:
1. Loading and preprocessing text8 data
2. Building vocabulary
3. Training Word2Vec model
4. Evaluating embeddings quality
5. Exporting model for production use
"""

import os
import sys
import argparse
import yaml
import pickle
import json
import logging
from typing import Dict, Any

import torch

from src.data_loader import prepare_text8_data, get_token_stream
from src.vocab import Vocabulary
from src.train import train_model
from src.utils import Timer, set_seed, export_model as utils_export_model
from src.evaluation import evaluate_embeddings


def setup_logging() -> None:
    """Configure logging for the process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('word2vec_pipeline.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Word2Vec Pipeline: Train, Evaluate, Export")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training (evaluate/export existing model)"
    )

    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation"
    )

    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip export"
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
        "eval_threshold": 0.0,  # No threshold by default
        "seed": 42
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            logging.info(f"Using default value for {key}: {value}")

    return config


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


def save_vocab_json(vocab: Vocabulary, output_path: str) -> None:
    """
    Save vocabulary to JSON format for inference.

    Args:
        vocab: Vocabulary object
        output_path: Path to save JSON vocabulary
    """
    vocab_export = {
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vocab_export, f, indent=2)

    logging.info(f"Saved vocabulary as JSON to {output_path}")


def main() -> None:
    """Main entry point for the pipeline."""
    # Setup
    setup_logging()
    args = parse_args()
    overall_timer = Timer()
    overall_timer.start()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_seed(config["seed"])

    # Prepare data and build vocabulary
    data_timer = Timer()
    data_timer.start()

    token_path = prepare_text8_data()
    vocab_path = config.get("vocab_path", "data/processed/vocab.pkl")
    vocab = build_or_load_vocab(token_path, vocab_path, config["min_count"])

    # Save vocabulary in JSON format for evaluation
    vocab_json_path = os.path.join(config["embeddings_dir"], "vocab.json")
    os.makedirs(config["embeddings_dir"], exist_ok=True)
    save_vocab_json(vocab, vocab_json_path)

    data_time = data_timer.elapsed()
    logging.info(f"Data preparation completed in {data_time:.2f}s")

    # Training
    model = None
    training_info = None
    train_time = 0

    if not args.skip_train:
        train_timer = Timer()
        train_timer.start()

        # Define token stream function for training
        def get_token_stream_fn():
            return get_token_stream(token_path)

        # Train model
        logging.info("Starting model training...")
        model, training_info = train_model(
            token_stream_fn=get_token_stream_fn,
            vocab=vocab,
            embed_dim=config["embed_dim"],
            window_size=config["window_size"],
            neg_samples=config["neg_samples"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            checkpoint_dir=config["checkpoint_dir"],
            embeddings_dir=config["embeddings_dir"],
            resume_from=args.resume,
            seed=config["seed"]
        )

        train_time = train_timer.elapsed()
        logging.info(f"Training completed in {train_time:.2f}s")

    # Evaluation
    eval_results = None
    eval_time = 0

    if not args.skip_eval:
        eval_timer = Timer()
        eval_timer.start()

        # Get path to embeddings
        embeddings_path = os.path.join(
            config["embeddings_dir"], "word_vectors.npy")

        # Ensure the path exists
        if not os.path.exists(embeddings_path):
            embeddings_path = os.path.join(
                config["embeddings_dir"], "word_vectors_final.npy")
            if not os.path.exists(embeddings_path):
                logging.error("No embeddings found for evaluation")
                if not args.skip_export:
                    logging.warning(
                        "Skipping export due to missing embeddings")
                    args.skip_export = True

        # Run evaluation if embeddings exist
        if os.path.exists(embeddings_path):
            logging.info("Evaluating model quality...")
            eval_output_path = os.path.join(
                config["embeddings_dir"], "evaluation.json")

            # Run evaluation
            eval_results = evaluate_embeddings(
                embeddings_path=embeddings_path,
                vocab_path=vocab_json_path,
                output_path=eval_output_path
            )

            eval_time = eval_timer.elapsed()
            logging.info(f"Evaluation completed in {eval_time:.2f}s")

            # Report scores
            combined_score = eval_results["overall_scores"]["combined_score"]
            logging.info(f"Model evaluation score: {combined_score:.4f}")

            # Check against threshold
            if (config["eval_threshold"] > 0 and
                    combined_score < config["eval_threshold"]):
                logging.warning(
                    "Model quality below threshold: "
                    f"{combined_score:.4f} < {config['eval_threshold']:.4f}")
                if not args.skip_export:
                    logging.warning("Skipping export due to low quality")
                    args.skip_export = True

    # Export
    export_time = 0
    export_paths = {}

    if not args.skip_export:
        export_timer = Timer()
        export_timer.start()

        export_dir = config.get("export_dir", "exported_model")

        # Check if we have a model object, if not we need to load it
        if model is None:
            # Find the best checkpoint
            best_checkpoint = None
            if os.path.exists(config["checkpoint_dir"]):
                for filename in os.listdir(config["checkpoint_dir"]):
                    if filename.endswith(".pt"):
                        best_checkpoint = os.path.join(
                            config["checkpoint_dir"], filename)
                        break

            if best_checkpoint:
                from src.model import SkipGramModel
                from src.utils import load_checkpoint

                # Load model from checkpoint
                logging.info(
                    f"Loading model from checkpoint: {best_checkpoint}")
                model = SkipGramModel(len(vocab), config["embed_dim"])
                optimizer = torch.optim.Adam(model.parameters())
                model, _, _, _ = load_checkpoint(
                    model, optimizer, best_checkpoint)
            else:
                logging.error("No model available for export")
                sys.exit(1)

        # Export model
        export_paths = utils_export_model(
            model,
            vocab,
            config,
            export_dir,
            args.version
        )

        # Add evaluation results if available
        if eval_results:
            eval_path = os.path.join(export_dir, "evaluation.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)

            export_paths["evaluation"] = eval_path
            logging.info(f"Saved evaluation results to {eval_path}")

        export_time = export_timer.elapsed()
        logging.info(f"Export completed in {export_time:.2f}s")
        logging.info(f"Model exported to {export_dir}")

    # Print final report
    total_time = overall_timer.elapsed()

    print("\n=== Word2Vec Pipeline Complete ===")
    print(f"Total time: {total_time:.2f}s")

    if not args.skip_train:
        print(f"- Data preparation: {data_time:.2f}s")
        print(f"- Training: {train_time:.2f}s")

    if not args.skip_eval and eval_results:
        print(f"- Evaluation: {eval_time:.2f}s")
        print(
            "- Evaluation score: "
            f"{eval_results['overall_scores']['combined_score']:.4f}")

    if not args.skip_export:
        print(f"- Export: {export_time:.2f}s")
        print(f"- Model exported to: {export_dir}")

    print("\nOutput files:")
    print(f"- Vocabulary: {vocab_path}")

    if training_info and training_info.get('best_model_path'):
        print(f"- Best checkpoint: {training_info['best_model_path']}")

    if not args.skip_export:
        print("\nExported files:")
        for key, path in export_paths.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
