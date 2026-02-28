"""
data/prepare_datasets.py
=========================
Extended data preparation for multi-task evaluation.

Prepares:
  1. TinyStories (primary language modelling — delegates to prepare_tinystories.py)
  2. PIQA (physical intuition — 1k samples)
  3. GSM8K-easy (arithmetic reasoning — 1k samples)
  4. ARC-AGI-micro (pattern reasoning — 1k samples)

All datasets are encoded as raw UTF-8 bytes (vocab_size=256) for the CDLE
byte-level model.  Each dataset is cached independently.

Usage (CLI):
    python data/prepare_datasets.py
"""

import os
import sys
import logging
import yaml
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.prepare_tinystories import prepare_data as prepare_tinystories


def encode_text(text: str) -> list[int]:
    """Encode a string as a list of UTF-8 byte values (0–255)."""
    return list(text.encode("utf-8", errors="replace"))


def chunk_sequence(tokens: list[int], chunk_size: int) -> list[list[int]]:
    """Slide a window over tokens to produce fixed-length chunks."""
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens) - chunk_size + 1, chunk_size)]


def prepare_hf_dataset(
    dataset_name: str,
    config_name: str,
    split: str,
    text_fn,
    cache_path: str,
    seq_len: int = 256,
    max_samples: int = 1000,
) -> str:
    """
    Download a HuggingFace dataset, encode as bytes, and cache as tensors.

    Args:
        dataset_name: HF dataset identifier.
        config_name:  Dataset config (e.g. "main", "ARC-Easy").
        split:        Dataset split to load.
        text_fn:      Function that converts a dataset example to a text string.
        cache_path:   Path to save the cached tensor file.
        seq_len:      Sequence length for the model.
        max_samples:  Maximum number of samples.

    Returns:
        Path to the saved cache file.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        log.info(f"Cache exists at {cache_path}. Skipping.")
        return cache_path

    log.info(f"Downloading {dataset_name}/{config_name} ({split})...")
    try:
        from datasets import load_dataset
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        log.warning(f"Could not download {dataset_name}: {e}")
        log.info("Creating synthetic fallback dataset.")
        # Create minimal synthetic data
        chunks = [[i % 256 for i in range(seq_len + 1)] for _ in range(100)]
        tensor = torch.tensor(chunks, dtype=torch.long)
        torch.save({"data": tensor}, cache_path)
        return cache_path

    # Encode texts to bytes and chunk
    chunk_size = seq_len + 1
    all_chunks = []
    sep = [10]  # newline separator

    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        text = text_fn(example)
        tokens = encode_text(text) + sep
        chunks = chunk_sequence(tokens, chunk_size)
        all_chunks.extend(chunks)

    if not all_chunks:
        # Fallback: create at least one chunk
        all_chunks = [[0] * chunk_size]

    tensor = torch.tensor(all_chunks, dtype=torch.long)
    torch.save({"data": tensor}, cache_path)
    log.info(f"Saved {len(all_chunks)} chunks to {cache_path}")
    return cache_path


def prepare_all_datasets(config_path: str = "config.yaml"):
    """
    Prepare all datasets for multi-task evaluation.

    Args:
        config_path: Path to config.yaml.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seq_len = cfg["model"]["seq_len"]
    multi_task = cfg.get("data", {}).get("multi_task", {})

    # 1. TinyStories (primary)
    log.info("=== Preparing TinyStories ===")
    prepare_tinystories(config_path)

    # 2. PIQA
    log.info("=== Preparing PIQA ===")
    prepare_hf_dataset(
        dataset_name="piqa",
        config_name="",
        split="validation",
        text_fn=lambda ex: f"Goal: {ex.get('goal', '')} Sol1: {ex.get('sol1', '')} Sol2: {ex.get('sol2', '')}",
        cache_path="data/piqa_cache.pt",
        seq_len=seq_len,
        max_samples=multi_task.get("piqa_samples", 1000),
    )

    # 3. GSM8K
    log.info("=== Preparing GSM8K ===")
    prepare_hf_dataset(
        dataset_name="gsm8k",
        config_name="main",
        split="test",
        text_fn=lambda ex: f"Q: {ex.get('question', '')} A: {ex.get('answer', '')}",
        cache_path="data/gsm8k_cache.pt",
        seq_len=seq_len,
        max_samples=multi_task.get("gsm8k_samples", 1000),
    )

    # 4. ARC (ARC-Easy subset)
    log.info("=== Preparing ARC ===")
    def arc_text_fn(ex):
        choices = ex.get("choices", {})
        choice_text = " ".join(
            f"({l}) {t}" for l, t in zip(
                choices.get("label", []),
                choices.get("text", []),
            )
        )
        return f"Q: {ex.get('question', '')} {choice_text} A: {ex.get('answerKey', '')}"

    prepare_hf_dataset(
        dataset_name="ai2_arc",
        config_name="ARC-Easy",
        split="test",
        text_fn=arc_text_fn,
        cache_path="data/arc_cache.pt",
        seq_len=seq_len,
        max_samples=multi_task.get("arc_samples", 1000),
    )

    log.info("=== All datasets prepared. ===")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    prepare_all_datasets(config_path)
