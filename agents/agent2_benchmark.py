"""
agents/agent2_benchmark.py
===========================
Agent 2: The Benchmarker

Role:
  Train both the CDLE model and the baseline transformer on TinyStories
  (10 k samples) and record performance metrics.

Pipeline:
  1. Load config.yaml.
  2. Prepare (or load cached) TinyStories data.
  3. Train CDLE model for `max_steps` steps, measuring wall-clock time.
  4. Train baseline transformer for the same number of steps.
  5. Compute val loss, train time, loss_per_watt, FLOPs estimate, param count.
  6. Write benchmark_results.json.

All training is CPU-only, PyTorch 2.x.

The script is called from GitHub Actions (benchmark.yml) but also runs
standalone: `python agents/agent2_benchmark.py`
"""

import os
import sys
import json
import math
import time
import logging
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cdle_base import CDLEModel
from models.baseline_transformer import BaselineTransformer
from data.prepare_tinystories import prepare_data
from utils.metrics import compute_loss_per_watt, estimate_flops, format_results, Timer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
RESULTS_PATH = "benchmark_results.json"


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def load_datasets(cache_path: str):
    """
    Load the pre-processed TinyStories tensors from disk.

    Returns:
        (train_data, val_data) — TensorDatasets with (input, target) pairs.
    """
    cache = torch.load(cache_path, weights_only=True)
    train_chunks = cache["train"]   # (N, seq_len + 1)
    val_chunks = cache["val"]       # (M, seq_len + 1)

    train_ds = TensorDataset(train_chunks[:, :-1], train_chunks[:, 1:])
    val_ds = TensorDataset(val_chunks[:, :-1], val_chunks[:, 1:])
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    cfg: dict,
    model_name: str,
) -> dict:
    """
    Train a model and return benchmark results.

    Args:
        model:      Untrained model instance.
        train_ds:   Training TensorDataset.
        val_ds:     Validation TensorDataset.
        cfg:        Full config dict.
        model_name: Identifier string for logging.

    Returns:
        Dict of benchmark metrics.
    """
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]

    batch_size: int = t_cfg["batch_size"]
    max_steps: int = t_cfg["max_steps"]
    lr: float = t_cfg["learning_rate"]
    wd: float = t_cfg["weight_decay"]
    grad_clip: float = t_cfg["grad_clip"]
    warmup_steps: int = t_cfg.get("warmup_steps", 50)
    val_every: int = t_cfg.get("val_every", 100)
    val_steps: int = t_cfg.get("val_steps", 20)

    model_type = "cdle" if "cdle" in model_name.lower() else "transformer"

    # Optimiser: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )

    # LR schedule: linear warmup → cosine decay
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.train()
    step = 0
    train_loss_accum = 0.0
    train_loss_count = 0
    last_val_loss = float("inf")

    log.info(
        f"[{model_name}] Starting training: {max_steps} steps, "
        f"batch={batch_size}, lr={lr}, params={model.count_parameters():,}"
    )

    train_start = time.time()

    # Infinite data loop
    data_iter = iter(train_loader)

    for step in range(max_steps):
        # Fetch next batch (restart loader if exhausted)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        # Forward pass
        logits, ff_loss = model(x, compute_ff_loss=True)

        # Primary loss: cross-entropy on next-token prediction
        # logits: (B, L, V) → (B*L, V), y: (B, L) → (B*L,)
        B, L, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.view(B * L, V),
            y.view(B * L),
        )

        # Combine CE loss with FF regularisation (small weight)
        total_loss = ce_loss
        if ff_loss is not None:
            total_loss = ce_loss + 0.01 * ff_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        train_loss_accum += ce_loss.item()
        train_loss_count += 1

        # Logging
        if (step + 1) % 50 == 0:
            avg_train_loss = train_loss_accum / train_loss_count
            elapsed = time.time() - train_start
            log.info(
                f"[{model_name}] step {step+1}/{max_steps} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"elapsed={elapsed:.1f}s"
            )
            train_loss_accum = 0.0
            train_loss_count = 0

        # Validation
        if (step + 1) % val_every == 0:
            last_val_loss = evaluate(model, val_loader, val_steps)
            log.info(f"[{model_name}] val_loss={last_val_loss:.4f}")
            model.train()

    # Final validation after training
    last_val_loss = evaluate(model, val_loader, val_steps)
    train_time = time.time() - train_start

    log.info(
        f"[{model_name}] Training complete: val_loss={last_val_loss:.4f}, "
        f"time={train_time:.1f}s"
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    lpw = compute_loss_per_watt(last_val_loss, train_time)
    flops = estimate_flops(
        d_model=m_cfg["d_model"],
        n_layers=m_cfg["n_layers"],
        d_state=m_cfg.get("d_state", 16),
        seq_len=m_cfg["seq_len"],
        batch_size=batch_size,
        model_type=model_type,
    )

    results = {
        "model_name": model_name,
        "val_loss": round(last_val_loss, 6),
        "train_time_s": round(train_time, 2),
        "loss_per_watt": round(lpw, 8),
        "flops_estimate": int(flops),
        "param_count": model.count_parameters(),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    val_steps: int,
) -> float:
    """
    Run validation and return mean cross-entropy loss.

    Args:
        model:      Model in eval mode (will be set to eval internally).
        val_loader: Validation DataLoader.
        val_steps:  Number of batches to evaluate.

    Returns:
        Mean validation cross-entropy loss (float).
    """
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= val_steps:
            break
        logits, _ = model(x, compute_ff_loss=False)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B * L, V), y.view(B * L))
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Agent 2: Benchmarker ===")

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Prepare data
    cache_path = cfg["data"]["cache_path"]
    if not os.path.exists(cache_path):
        log.info("Data cache not found. Running prepare_data…")
        prepare_data(CONFIG_PATH)

    train_ds, val_ds = load_datasets(cache_path)
    log.info(
        f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val chunks."
    )

    all_results = {}

    # ------------------------------------------------------------------
    # Benchmark CDLE model
    # ------------------------------------------------------------------
    cdle_model = CDLEModel.from_config(cfg)
    log.info(f"CDLE model params: {cdle_model.count_parameters():,}")

    cdle_results = train_model(cdle_model, train_ds, val_ds, cfg, "CDLE")
    all_results["cdle"] = cdle_results
    print(format_results(cdle_results, "CDLE Benchmark"))

    # ------------------------------------------------------------------
    # Benchmark Baseline Transformer
    # ------------------------------------------------------------------
    baseline_model = BaselineTransformer.from_config(cfg)
    log.info(f"Baseline model params: {baseline_model.count_parameters():,}")

    baseline_results = train_model(
        baseline_model, train_ds, val_ds, cfg, "BaselineTransformer"
    )
    all_results["baseline"] = baseline_results
    print(format_results(baseline_results, "Baseline Benchmark"))

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    cdle_lpw = cdle_results["loss_per_watt"]
    base_lpw = baseline_results["loss_per_watt"]
    improvement = (base_lpw - cdle_lpw) / max(base_lpw, 1e-9) * 100
    all_results["summary"] = {
        "cdle_vs_baseline_lpw_improvement_pct": round(improvement, 2),
        "winner": "cdle" if cdle_lpw < base_lpw else "baseline",
    }

    log.info(
        f"CDLE loss_per_watt: {cdle_lpw:.6f} | "
        f"Baseline: {base_lpw:.6f} | "
        f"CDLE improvement: {improvement:+.1f}%"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Benchmark results saved to {RESULTS_PATH}")
    log.info("=== Agent 2 complete. ===")


if __name__ == "__main__":
    main()
