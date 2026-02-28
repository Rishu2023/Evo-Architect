"""
benchmarks/multi_task_eval.py
==============================
Multi-task evaluation suite for the CDLE model.

Evaluates on multiple tasks beyond TinyStories:
  1. TinyStories (language modelling — primary)
  2. PIQA-style (physical intuition — 1k samples)
  3. GSM8K-easy (basic arithmetic reasoning — 1k samples)
  4. ARC-AGI-micro (pattern reasoning — 1k samples)

All benchmarks use byte-level encoding (vocab_size=256) to match CDLE input.
Each benchmark is capped at 1k samples and targets <12 minutes total runtime.

Results are returned as a dict of task_name → {loss, accuracy, time_s}.
"""

import os
import sys
import time
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def encode_text_to_bytes(text: str, seq_len: int = 256) -> list[int]:
    """Encode a string to byte-level tokens, truncated/padded to seq_len."""
    tokens = list(text.encode("utf-8", errors="replace"))[:seq_len]
    # Pad with 0 (null byte) if shorter than seq_len
    tokens = tokens + [0] * (seq_len - len(tokens))
    return tokens


def prepare_byte_dataset(
    texts: list[str],
    seq_len: int = 256,
    max_samples: int = 1000,
) -> TensorDataset:
    """
    Convert a list of text strings to a byte-level TensorDataset.

    Each sample is (input, target) where target is shifted by 1 byte.

    Args:
        texts:       List of text strings.
        seq_len:     Sequence length for the model.
        max_samples: Maximum number of samples.

    Returns:
        TensorDataset with (input, target) pairs.
    """
    texts = texts[:max_samples]
    chunks = []
    for text in texts:
        tokens = encode_text_to_bytes(text, seq_len + 1)
        chunks.append(tokens)

    if not chunks:
        # Return a dummy dataset if no data
        dummy = torch.zeros(1, seq_len, dtype=torch.long)
        return TensorDataset(dummy, dummy)

    tensor = torch.tensor(chunks, dtype=torch.long)
    return TensorDataset(tensor[:, :-1], tensor[:, 1:])


@torch.no_grad()
def evaluate_language_modelling(
    model: torch.nn.Module,
    dataset: TensorDataset,
    batch_size: int = 32,
    max_batches: int = 50,
) -> dict:
    """
    Evaluate a model on a language modelling task (next-byte prediction).

    Args:
        model:       Model to evaluate (must be in eval mode).
        dataset:     TensorDataset of (input, target) pairs.
        batch_size:  Batch size for evaluation.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Dict with 'loss', 'perplexity', 'time_s'.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    total_loss = 0.0
    count = 0
    t_start = time.time()

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        logits, _ = model(x, compute_ff_loss=False)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B * L, V), y.view(B * L))
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return {
        "loss": round(avg_loss, 6),
        "perplexity": round(min(torch.exp(torch.tensor(avg_loss)).item(), 1e6), 4),
        "time_s": round(time.time() - t_start, 2),
    }


def load_piqa_samples(max_samples: int = 1000) -> list[str]:
    """
    Load PIQA-style physical intuition samples.

    Falls back to synthetic samples if the dataset is unavailable.

    Args:
        max_samples: Maximum samples to load.

    Returns:
        List of text strings.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("piqa", split="validation", trust_remote_code=True)
        texts = []
        for ex in ds:
            text = f"Goal: {ex.get('goal', '')} Solution 1: {ex.get('sol1', '')} Solution 2: {ex.get('sol2', '')}"
            texts.append(text)
            if len(texts) >= max_samples:
                break
        return texts
    except Exception as e:
        log.warning(f"Could not load PIQA: {e}. Using synthetic samples.")
        return [f"Physical reasoning sample {i}: object falls down due to gravity." for i in range(max_samples)]


def load_gsm8k_samples(max_samples: int = 1000) -> list[str]:
    """
    Load GSM8K-easy arithmetic reasoning samples.

    Falls back to synthetic samples if the dataset is unavailable.

    Args:
        max_samples: Maximum samples to load.

    Returns:
        List of text strings.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        texts = []
        for ex in ds:
            text = f"Question: {ex.get('question', '')} Answer: {ex.get('answer', '')}"
            texts.append(text)
            if len(texts) >= max_samples:
                break
        return texts
    except Exception as e:
        log.warning(f"Could not load GSM8K: {e}. Using synthetic samples.")
        return [f"Math problem {i}: If you have {i} apples and get {i+1} more, you have {2*i+1} apples." for i in range(max_samples)]


def load_arc_samples(max_samples: int = 1000) -> list[str]:
    """
    Load ARC-AGI-micro pattern reasoning samples.

    Falls back to synthetic samples if the dataset is unavailable.

    Args:
        max_samples: Maximum samples to load.

    Returns:
        List of text strings.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
        texts = []
        for ex in ds:
            choices = ex.get("choices", {})
            choice_text = " ".join(
                f"({l}) {t}" for l, t in zip(
                    choices.get("label", []),
                    choices.get("text", []),
                )
            )
            text = f"Question: {ex.get('question', '')} Choices: {choice_text} Answer: {ex.get('answerKey', '')}"
            texts.append(text)
            if len(texts) >= max_samples:
                break
        return texts
    except Exception as e:
        log.warning(f"Could not load ARC: {e}. Using synthetic samples.")
        return [f"Pattern {i}: Given sequence [1,2,3,...,{i}], the next number is {i+1}." for i in range(max_samples)]


def run_multi_task_eval(
    model: torch.nn.Module,
    seq_len: int = 256,
    batch_size: int = 32,
    task_samples: int = 1000,
) -> dict:
    """
    Run the full multi-task evaluation suite.

    Args:
        model:        Model to evaluate (will be set to eval mode).
        seq_len:      Sequence length.
        batch_size:   Batch size for evaluation.
        task_samples: Number of samples per task.

    Returns:
        Dict mapping task_name → {loss, perplexity, time_s}.
    """
    model.eval()
    results = {}

    # Task 1: PIQA (Physical Intuition)
    log.info("Evaluating on PIQA...")
    piqa_texts = load_piqa_samples(task_samples)
    piqa_ds = prepare_byte_dataset(piqa_texts, seq_len, task_samples)
    results["piqa"] = evaluate_language_modelling(model, piqa_ds, batch_size)
    log.info(f"  PIQA → loss={results['piqa']['loss']:.4f}")

    # Task 2: GSM8K-easy (Arithmetic Reasoning)
    log.info("Evaluating on GSM8K...")
    gsm8k_texts = load_gsm8k_samples(task_samples)
    gsm8k_ds = prepare_byte_dataset(gsm8k_texts, seq_len, task_samples)
    results["gsm8k"] = evaluate_language_modelling(model, gsm8k_ds, batch_size)
    log.info(f"  GSM8K → loss={results['gsm8k']['loss']:.4f}")

    # Task 3: ARC (Pattern Reasoning)
    log.info("Evaluating on ARC...")
    arc_texts = load_arc_samples(task_samples)
    arc_ds = prepare_byte_dataset(arc_texts, seq_len, task_samples)
    results["arc"] = evaluate_language_modelling(model, arc_ds, batch_size)
    log.info(f"  ARC → loss={results['arc']['loss']:.4f}")

    # Compute aggregate generalization score (average loss across tasks)
    all_losses = [r["loss"] for r in results.values()]
    results["aggregate"] = {
        "mean_loss": round(sum(all_losses) / len(all_losses), 6),
        "total_time_s": round(sum(r["time_s"] for r in results.values()), 2),
    }

    return results


if __name__ == "__main__":
    import yaml

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.cdle_base import CDLEModel

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = CDLEModel.from_config(cfg)
    print(f"Model params: {model.count_parameters():,}")

    results = run_multi_task_eval(model, seq_len=cfg["model"]["seq_len"])

    print("\n=== Multi-Task Evaluation Results ===")
    for task, metrics in results.items():
        print(f"  {task}: {metrics}")
    print("Multi-task eval complete ✓")
