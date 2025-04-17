#!/usr/bin/env python
"""
Train Word2Vec model on the text8 dataset from Hugging Face
"""

import os
import argparse
import yaml
import pickle
from datasets import load_dataset

from src.tokenize import load_and_tokenize_text8
from src.vocab import Vocabulary
from src.train import train_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on text8 dataset")
    parser.add_argument("--config", type=str, default="train_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_to_file(tokens, output_path):
    """Save tokens to a file, one sentence per line"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert the tokens to a single line of space-separated tokens
        f.write(" ".join(tokens))
    print(f"Saved tokens to {output_path}")


def prepare_text8_data():
    """
    Download and prepare the text8 dataset

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
        print(f"Using existing tokenized data at {token_path}")
        return token_path

    # Download and process the dataset
    print("Downloading text8 dataset from Hugging Face...")
    dataset = load_dataset("afmck/text8")

    # Tokenize the data
    print("Tokenizing data...")
    tokens = load_and_tokenize_text8(dataset)

    # Save tokens to file
    save_to_file(tokens, token_path)

    return token_path


def build_or_load_vocab(token_path, vocab_path, min_freq):
    """
    Build vocabulary from tokens or load from file if it exists

    Args:
        token_path: Path to tokenized data
        vocab_path: Path to save/load vocabulary
        min_freq: Minimum word frequency

    Returns:
        Vocabulary: Vocabulary object
    """
    # Check if vocabulary exists
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    # Create vocabulary
    print(f"Building vocabulary with min_freq={min_freq}...")

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

    print(f"Vocabulary built with {len(vocab)} words")
    print(f"Saved vocabulary to {vocab_path}")

    return vocab


def main():
    """Main entry point"""
    # Parse arguments and load configuration
    args = parse_args()
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

    # Train model
    train_model(
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
        resume_from=args.resume,
        seed=config["seed"]
    )


if __name__ == "__main__":
    main()
