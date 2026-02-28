"""
data/prepare_tinystories.py
============================
Download the first 10 000 samples from the TinyStories dataset (Hugging Face)
and save them as a PyTorch-compatible cache for fast reloading during training.

Usage (CLI):
    python data/prepare_tinystories.py

The script:
  1. Downloads roneneldan/TinyStories via the `datasets` library.
  2. Encodes each story as raw UTF-8 bytes (vocab_size = 256, no tokeniser).
  3. Splits into train (9 000) and validation (1 000) sets.
  4. Saves a dict of tensors to data/tinystories_cache.pt.

Data format saved:
  {
    "train": LongTensor of shape (num_train_chunks, seq_len + 1),
    "val":   LongTensor of shape (num_val_chunks,   seq_len + 1),
  }

Each row is a context+target pair of length (seq_len + 1):
  input  = row[:-1]   (seq_len tokens)
  target = row[1:]    (seq_len tokens, shifted by 1 for next-token prediction)
"""

import os
import sys
import logging
import yaml
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def encode_text(text: str) -> list[int]:
    """Encode a string as a list of UTF-8 byte values (0–255)."""
    return list(text.encode("utf-8", errors="replace"))


def chunk_sequence(
    tokens: list[int],
    chunk_size: int,
    stride: int,
) -> list[list[int]]:
    """
    Slide a window over a token list to produce fixed-length chunks.

    Args:
        tokens:     Flat list of integer token IDs.
        chunk_size: Length of each chunk (seq_len + 1).
        stride:     Step between chunk starts (= chunk_size for non-overlapping).

    Returns:
        List of chunks, each of length chunk_size.
    """
    chunks = []
    for start in range(0, len(tokens) - chunk_size + 1, stride):
        chunks.append(tokens[start : start + chunk_size])
    return chunks


def prepare_data(config_path: str = "config.yaml") -> str:
    """
    Main data preparation routine.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Path to the saved cache file.
    """
    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    seq_len: int = model_cfg["seq_len"]
    num_train: int = data_cfg["num_train_samples"]
    num_val: int = data_cfg["num_val_samples"]
    cache_path: str = data_cfg["cache_path"]
    dataset_name: str = data_cfg["dataset_name"]

    # ------------------------------------------------------------------
    # Check cache
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        log.info(f"Cache already exists at {cache_path}. Loading.")
        cache = torch.load(cache_path, weights_only=True)
        log.info(
            f"Train chunks: {len(cache['train'])}, Val chunks: {len(cache['val'])}"
        )
        return cache_path

    # ------------------------------------------------------------------
    # Download dataset
    # ------------------------------------------------------------------
    log.info(f"Downloading {dataset_name} (first {num_train + num_val} samples)…")
    from datasets import load_dataset  # lazy import to speed up --help

    # Use streaming=False so we can index; only load the first split
    dataset = load_dataset(
        dataset_name,
        split="train",
        trust_remote_code=True,
    )

    total_needed = num_train + num_val
    dataset = dataset.select(range(min(total_needed, len(dataset))))
    log.info(f"Loaded {len(dataset)} stories from {dataset_name}.")

    # ------------------------------------------------------------------
    # Encode all stories → flat byte sequences → chunks
    # ------------------------------------------------------------------
    chunk_size = seq_len + 1  # +1 because we need input+target
    train_chunks: list[list[int]] = []
    val_chunks: list[list[int]] = []

    # Separator byte between stories (newline = 10)
    sep = [10]

    for i, example in enumerate(dataset):
        text: str = example.get("text", example.get("story", ""))
        tokens = encode_text(text) + sep
        chunks = chunk_sequence(tokens, chunk_size, stride=chunk_size)

        if i < num_train:
            train_chunks.extend(chunks)
        else:
            val_chunks.extend(chunks)

    log.info(
        f"Chunks — train: {len(train_chunks)}, val: {len(val_chunks)}"
    )

    if not train_chunks:
        raise RuntimeError(
            "No training chunks created. Check seq_len vs. average story length."
        )

    # ------------------------------------------------------------------
    # Convert to tensors
    # ------------------------------------------------------------------
    train_tensor = torch.tensor(train_chunks, dtype=torch.long)
    val_tensor = torch.tensor(val_chunks, dtype=torch.long) if val_chunks else train_tensor[:100]

    # ------------------------------------------------------------------
    # Save cache
    # ------------------------------------------------------------------
    cache = {"train": train_tensor, "val": val_tensor}
    torch.save(cache, cache_path)
    log.info(f"Saved cache to {cache_path}")
    log.info(f"  train shape: {train_tensor.shape}")
    log.info(f"  val   shape: {val_tensor.shape}")
    return cache_path


if __name__ == "__main__":
    # Support optional config path as CLI argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    prepare_data(config_path)
