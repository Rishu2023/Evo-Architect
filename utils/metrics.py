"""
utils/metrics.py
================
Shared metrics utilities used by Agent 2 (benchmarker) and Agent 3 (router).

Provides:
  - compute_loss_per_watt: main efficiency metric
  - estimate_flops: rough FLOPs estimate for a forward pass
  - format_results: pretty-print a results dict
  - pareto_score / is_pareto_dominated: Pareto multi-objective scoring
  - estimate_energy: energy proxy utilities
  - compute_qd_niche / update_qd_archive: Quality-Diversity (MAP-Elites) helpers
  - compute_sparsity: fraction of near-zero weights
  - estimate_memory_mb: model memory estimate
"""

import time
from typing import Dict, Any, Optional, List, Tuple

import torch


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


# ---------------------------------------------------------------------------
# Pareto Multi-Objective Scoring
# ---------------------------------------------------------------------------

# Default weights for each objective (lower is better for all).
_PARETO_WEIGHTS: Dict[str, float] = {
    "loss_per_sec": 0.30,
    "sparsity": 0.15,
    "memory": 0.15,
    "generalization": 0.25,
    "continual_adaptation_delta": 0.15,
}


def pareto_score(objectives: dict) -> float:
    """
    Compute a weighted Pareto score from multiple objectives.

    Each objective value is multiplied by its weight and summed.
    Missing objectives are skipped (their weight is redistributed
    proportionally among the objectives that are present).

    Lower is better — all constituent objectives should follow the
    "lower is better" convention.

    Args:
        objectives: Mapping of objective name to its numeric value.
                    Recognised keys: loss_per_sec, sparsity, memory,
                    generalization, continual_adaptation_delta.

    Returns:
        Weighted composite score (float).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for key, weight in _PARETO_WEIGHTS.items():
        if key in objectives:
            weighted_sum += weight * objectives[key]
            total_weight += weight

    if total_weight == 0.0:
        return float("inf")

    # Normalise so that the weights of present objectives sum to 1.
    return weighted_sum / total_weight


def is_pareto_dominated(candidate: dict, archive: List[dict]) -> bool:
    """
    Check whether *candidate* is Pareto-dominated by any member of *archive*.

    A candidate is dominated if there exists an archive member that is at
    least as good on **every** objective and strictly better on at least one.

    Both *candidate* and each archive member must be dicts whose keys are
    a subset of the recognised objective names.  Only objectives present in
    **both** the candidate and the archive member are compared.

    Args:
        candidate: Dict of objective name → value for the candidate.
        archive:   List of dicts with the same structure.

    Returns:
        True if *candidate* is dominated by at least one archive member.
    """
    obj_keys = list(_PARETO_WEIGHTS.keys())

    for member in archive:
        # Only compare on objectives shared by both.
        shared = [k for k in obj_keys if k in candidate and k in member]
        if not shared:
            continue

        # "at least as good on all" and "strictly better on at least one"
        all_leq = all(member[k] <= candidate[k] for k in shared)
        any_lt = any(member[k] < candidate[k] for k in shared)

        if all_leq and any_lt:
            return True

    return False


# ---------------------------------------------------------------------------
# Energy Proxy Utilities
# ---------------------------------------------------------------------------


def estimate_energy(
    flops: float,
    wall_time_s: float,
    cpu_tdp_watts: float = 65.0,
) -> dict:
    """
    Estimate energy consumption from FLOPs and wall-clock time.

    Provides a simple linear energy model:
        linear_watt_estimate  ≈  wall_time_s  ×  cpu_tdp_watts   (Joules)
        energy_score = flops_per_sec / linear_watt_estimate

    A higher *energy_score* means more compute per joule (better).
    Note: *linear_watt_estimate* is energy in joules (time × power).

    Args:
        flops:          Total floating-point operations performed.
        wall_time_s:    Wall-clock time in seconds.
        cpu_tdp_watts:  Thermal-design power of the processor (default 65 W).

    Returns:
        Dict with keys: flops_per_sec, linear_watt_estimate, energy_score.
    """
    if wall_time_s <= 0:
        return {
            "flops_per_sec": float("inf"),
            "linear_watt_estimate": 0.0,
            "energy_score": float("inf"),
        }

    flops_per_sec = flops / wall_time_s
    # Simple linear energy model: energy (J) = time (s) × power (W).
    # Named "linear_watt_estimate" per project convention; unit is joules.
    linear_watt_estimate = wall_time_s * cpu_tdp_watts

    # Avoid division by zero.
    energy_score = (
        flops_per_sec / linear_watt_estimate if linear_watt_estimate > 0 else 0.0
    )

    return {
        "flops_per_sec": flops_per_sec,
        "linear_watt_estimate": linear_watt_estimate,
        "energy_score": energy_score,
    }


# ---------------------------------------------------------------------------
# Quality-Diversity (MAP-Elites) Utilities
# ---------------------------------------------------------------------------


def compute_qd_niche(
    complexity: float,
    sparsity: float,
    n_complexity_bins: int = 4,
    n_sparsity_bins: int = 3,
) -> Tuple[int, int]:
    """
    Return the MAP-Elites grid cell indices for a candidate.

    Both *complexity* and *sparsity* are expected to lie in [0, 1].
    Values outside this range are clamped.

    Args:
        complexity:       Normalised complexity descriptor in [0, 1].
        sparsity:         Normalised sparsity descriptor in [0, 1].
        n_complexity_bins: Number of bins along the complexity axis.
        n_sparsity_bins:   Number of bins along the sparsity axis.

    Returns:
        (complexity_bin, sparsity_bin) tuple of integer indices.
    """
    # Clamp to [0, 1] then discretise.
    complexity = max(0.0, min(1.0, complexity))
    sparsity = max(0.0, min(1.0, sparsity))

    c_bin = min(int(complexity * n_complexity_bins), n_complexity_bins - 1)
    s_bin = min(int(sparsity * n_sparsity_bins), n_sparsity_bins - 1)
    return (c_bin, s_bin)


def update_qd_archive(
    archive: dict,
    candidate: dict,
    niche: Tuple[int, int],
) -> bool:
    """
    Insert *candidate* into the QD archive at *niche* if the niche is empty
    or the candidate has a better (lower) fitness than the current occupant.

    The candidate dict must contain a ``"fitness"`` key.

    Args:
        archive:   Dict mapping niche tuples to candidate dicts.
        candidate: Dict with at least a ``"fitness"`` key (lower is better).
        niche:     (complexity_bin, sparsity_bin) tuple from compute_qd_niche.

    Returns:
        True if the candidate was inserted (archive was updated), else False.
    """
    current = archive.get(niche)
    if current is None or candidate["fitness"] < current["fitness"]:
        archive[niche] = candidate
        return True
    return False


# ---------------------------------------------------------------------------
# Sparsity Metric
# ---------------------------------------------------------------------------


def compute_sparsity(model: torch.nn.Module) -> float:
    """
    Compute the fraction of near-zero weights in a PyTorch model.

    A weight element is considered "near-zero" if its absolute value is
    less than 1e-4.

    Args:
        model: A ``torch.nn.Module`` whose parameters are inspected.

    Returns:
        Sparsity ratio in [0, 1].  Returns 0.0 if the model has no
        parameters.
    """
    total_params = 0
    near_zero = 0

    for param in model.parameters():
        total_params += param.numel()
        near_zero += int(torch.sum(torch.abs(param.data) < 1e-4).item())

    if total_params == 0:
        return 0.0
    return near_zero / total_params


# ---------------------------------------------------------------------------
# Memory Metric
# ---------------------------------------------------------------------------


def estimate_memory_mb(model: torch.nn.Module) -> float:
    """
    Estimate the memory footprint of a PyTorch model in megabytes.

    Sums ``param.nelement() * param.element_size()`` across all parameters
    and buffers, then converts bytes → MB (1 MB = 1 048 576 bytes).

    Args:
        model: A ``torch.nn.Module``.

    Returns:
        Estimated memory usage in MB.
    """
    total_bytes = 0

    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()

    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()

    return total_bytes / (1024 * 1024)
