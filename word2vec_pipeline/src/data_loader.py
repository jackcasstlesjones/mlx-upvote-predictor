"""
Data loading utilities for Word2Vec training.

This module provides functions to:
- Load and preprocess the text8 dataset
- Stream tokens efficiently for memory-friendly processing
- Save and load tokenized data
"""

import os
import logging
from typing import List, Generator, Dict, Any

from datasets import load_dataset

from .tokenize import tokenize


def load_text8_dataset() -> Dict[str, Any]:
    """
    Load text8 dataset from Hugging Face.

    Returns:
        Dict[str, Any]: Text8 dataset dictionary
    """
    logging.info("Downloading text8 dataset from Hugging Face...")
    return load_dataset("afmck/text8")


def tokenize_text8(dataset: Dict[str, Any]) -> List[str]:
    """
    Tokenize text8 dataset.

    Args:
        dataset: Text8 dataset from Hugging Face

    Returns:
        List[str]: Tokenized text
    """
    logging.info("Tokenizing text8 dataset...")
    text = dataset["train"]["text"][0]
    return tokenize(text)


def save_tokens(tokens: List[str], output_path: str) -> None:
    """
    Save tokens to file as space-separated text.

    Args:
        tokens: List of tokens
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info(f"Saving {len(tokens)} tokens to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))


def load_tokens(input_path: str) -> List[str]:
    """
    Load tokens from file.

    Args:
        input_path: Input file path

    Returns:
        List[str]: List of tokens
    """
    logging.info(f"Loading tokens from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read().split()


def get_token_stream(path: str) -> Generator[List[str], None, None]:
    """
    Create token stream generator from file.

    Args:
        path: Path to tokens file

    Yields:
        List[str]: Chunks of tokens
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def prepare_text8_data(output_dir: str = "data/processed") -> str:
    """
    Download, tokenize and save text8 dataset.

    Args:
        output_dir: Directory to save processed data

    Returns:
        str: Path to tokenized data file
    """
    # Define paths
    os.makedirs(output_dir, exist_ok=True)
    token_path = os.path.join(output_dir, "text8_tokens.txt")

    # Check if processed file already exists
    if os.path.exists(token_path):
        logging.info(f"Using existing tokenized data at {token_path}")
        return token_path

    # Download and tokenize
    dataset = load_text8_dataset()
    tokens = tokenize_text8(dataset)

    # Save tokens
    save_tokens(tokens, token_path)

    return token_path
