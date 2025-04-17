"""
Export Word2Vec model for production use.

This script converts the best trained Word2Vec model from checkpoint format
to a production-ready format with all necessary files for inference.
"""

from src.model import SkipGramModel
from src.utils import Timer, load_checkpoint
import os
import argparse
import json
import pickle
import logging
import sys
import time
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging() -> None:
    """Configure logging for the export process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('export.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Word2Vec model for production use")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file"
    )

    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary .pkl file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_model",
        help="Directory to save exported model"
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version string"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config file (optional)"
    )

    return parser.parse_args()


def load_model(
        checkpoint_path: str,
        vocab_size: int,
        embed_dim: int
) -> SkipGramModel:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension

    Returns:
        SkipGramModel: Loaded model
    """
    model = SkipGramModel(vocab_size, embed_dim)
    dummy_optimizer = torch.optim.Adam(model.parameters())

    model, _, epoch, loss = load_checkpoint(
        model, dummy_optimizer, checkpoint_path)
    model.eval()

    logging.info(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
    return model


def load_vocab(vocab_path: str) -> Any:
    """
    Load vocabulary from pickle file.

    Args:
        vocab_path: Path to vocabulary pickle file

    Returns:
        Any: Loaded vocabulary object
    """
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    logging.info(f"Loaded vocabulary with {len(vocab)} tokens")
    return vocab


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load training configuration.

    Args:
        config_path: Path to config file or None

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not config_path:
        # Default config
        return {
            "embed_dim": 100,
            "window_size": 5,
            "neg_samples": 10,
            "min_count": 5
        }

    # Load from file (either JSON or YAML)
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.warning(
            f"Unknown config format: {config_path}, using defaults")
        return {
            "embed_dim": 100,
            "window_size": 5,
            "neg_samples": 10,
            "min_count": 5
        }


def export_model_files(
    model: SkipGramModel,
    vocab: Any,
    config: Dict[str, Any],
    output_dir: str,
    version: str
) -> Dict[str, str]:
    """
    Export model files for production use.

    Creates:
    - model.pt: TorchScript model for inference
    - embeddings.npy: NumPy array of word embeddings
    - vocab.json: Vocabulary mapping in JSON format
    - metadata.json: Training parameters and configuration

    Args:
        model: Trained model
        vocab: Vocabulary object
        config: Training configuration
        output_dir: Output directory
        version: Model version string

    Returns:
        Dict[str, str]: Paths to exported files
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    model_path = os.path.join(output_dir, "model.pt")
    script_path = os.path.join(output_dir, "model_script.pt")
    embed_path = os.path.join(output_dir, "embeddings.npy")
    vocab_path = os.path.join(output_dir, "vocab.json")
    meta_path = os.path.join(output_dir, "metadata.json")

    # 1. Save PyTorch model
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model to {model_path}")

    # 2. Save TorchScript model (for inference without Python dependencies)
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, script_path)
        logging.info(f"Saved TorchScript model to {script_path}")
    except Exception as e:
        logging.warning(f"Failed to save TorchScript model: {e}")

    # 3. Save embeddings
    embeddings = model.get_in_embeddings()
    np.save(embed_path, embeddings)
    logging.info(f"Saved embeddings to {embed_path}")

    # 4. Save vocabulary mapping (simplified for inference)
    vocab_export = {
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_export, f, indent=2)
    logging.info(f"Saved vocabulary to {vocab_path}")

    # 5. Save metadata
    embed_dim = model.in_embed.embedding_dim
    metadata = {
        "version": version,
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": len(vocab),
        "embed_dim": embed_dim,
        "window_size": config.get("window_size", 5),
        "min_count": config.get("min_count", 5),
        "neg_samples": config.get("neg_samples", 10),
        "model_type": "SkipGram"
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {meta_path}")

    return {
        "model": model_path,
        "script": script_path,
        "embeddings": embed_path,
        "vocabulary": vocab_path,
        "metadata": meta_path
    }


def main() -> None:
    """Main entry point for export script."""
    # Setup
    setup_logging()
    args = parse_args()
    timer = Timer()
    timer.start()

    # Load vocabulary
    vocab = load_vocab(args.vocab)

    # Load configuration
    config = load_config(args.config)
    embed_dim = config.get("embed_dim", 100)

    # Load model
    model = load_model(args.checkpoint, len(vocab), embed_dim)

    # Export model
    output_paths = export_model_files(
        model,
        vocab,
        config,
        args.output_dir,
        args.version
    )

    # Report results
    elapsed = timer.elapsed()
    logging.info(f"Export completed in {elapsed:.2f}s")
    logging.info(f"Exported files: {output_paths}")

    print("\nExport successful! Files created:")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")
    print(f"\nTotal export time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
