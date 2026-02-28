"""
utils/metrics.py
================
Shared metrics utilities used by Agent 2 (benchmarker) and Agent 3 (router).

Provides:
  - compute_loss_per_watt: main efficiency metric
  - estimate_flops: rough FLOPs estimate for a forward pass
  - format_results: pretty-print a results dict
"""

import time
from typing import Dict, Any, Optional


def compute_loss_per_watt(
    val_loss: float,
    train_time_s: float,
) -> float:
    """
    Compute the primary evolutionary fitness metric.

    loss_per_watt = val_loss / train_time_s

    Lower is better: a model that achieves the same loss in less time is more
    efficient (using less compute/"wattage").

    Args:
        val_loss:     Final validation cross-entropy loss (nats).
        train_time_s: Total wall-clock training time in seconds.

    Returns:
        Efficiency score (lower is better).
    """
    if train_time_s <= 0:
        return float("inf")
    return val_loss / train_time_s


def estimate_flops(
    d_model: int,
    n_layers: int,
    d_state: int,
    seq_len: int,
    batch_size: int = 1,
    model_type: str = "cdle",
) -> float:
    """
    Estimate the number of floating-point operations for one forward pass.

    These are rough order-of-magnitude estimates — not exact counts.

    For CDLE (SSM-based):
      - SSM scan: O(L * D * N)  per layer   (no quadratic attention)
      - LTC + FF layers: O(L * D²) per layer

    For baseline transformer:
      - Attention: O(L² * D) per layer      (quadratic in sequence length)
      - MLP: O(L * D * d_ff) per layer

    Args:
        d_model:    Hidden dimension.
        n_layers:   Number of layers.
        d_state:    SSM state size (only relevant for CDLE).
        seq_len:    Sequence length.
        batch_size: Batch size.
        model_type: "cdle" or "transformer".

    Returns:
        Estimated FLOPs as a float.
    """
    if model_type == "cdle":
        # SSM scan dominates: 2 * L * D * N multiplications per layer
        ssm_flops = 2 * seq_len * d_model * d_state * n_layers
        # LTC + FF linear layers: 2 * L * D² per layer (2× for fwd+bwd-style)
        ltc_ff_flops = 4 * seq_len * d_model * d_model * n_layers
        total = (ssm_flops + ltc_ff_flops) * batch_size
    else:  # transformer
        # Attention: 4 * L² * D per layer
        attn_flops = 4 * seq_len * seq_len * d_model * n_layers
        # MLP (d_ff ≈ 2 * d_model): 4 * L * D * d_ff per layer
        d_ff = 2 * d_model
        mlp_flops = 4 * seq_len * d_model * d_ff * n_layers
        total = (attn_flops + mlp_flops) * batch_size

    return float(total)


def format_results(results: Dict[str, Any], title: str = "Benchmark Results") -> str:
    """
    Return a human-readable string representation of a results dictionary.

    Args:
        results: Dict of metric name → value.
        title:   Header string.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 60,
        f"  {title}",
        "=" * 60,
    ]
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key:<30s} {value:.6f}")
        elif isinstance(value, int):
            lines.append(f"  {key:<30s} {value:,}")
        else:
            lines.append(f"  {key:<30s} {value}")
    lines.append("=" * 60)
    return "\n".join(lines)


class Timer:
    """Simple context manager for measuring elapsed wall-clock time."""

    def __init__(self):
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
