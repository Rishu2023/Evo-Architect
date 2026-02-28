"""
agents/agent3_evolutionary_router.py
======================================
Agent 3: The Evolutionary Router

Role:
  Compare the latest benchmark results against the evolutionary memory and
  decide whether the new architecture should be:
    - **merged**: it outperforms the current best → update memory, keep config
    - **discarded**: it performs worse → revert to best known config

Algorithm:
  1. Load benchmark_results.json (written by Agent 2).
  2. Load evolutionary_memory.json.
  3. Compare new CDLE `loss_per_watt` against `memory["best_loss_per_watt"]`.
  4. If improvement exceeds `improvement_threshold`:
       - Mark verdict = "merged"
       - Update best metrics in memory
  5. Else:
       - Mark verdict = "discarded"
       - Restore best_config to active_config
  6. Append generation record to history.
  7. Save updated memory.
  8. Exit with code 0 (merged) or 1 (discarded) so the CI workflow can branch.

The improvement_threshold (from config.yaml) prevents noise-driven merges.
"""

import os
import sys
import json
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = "config.yaml"
MEMORY_PATH = "evolutionary_memory.json"
RESULTS_PATH = "benchmark_results.json"


def load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str) -> None:
    """Save a dict as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved → {path}")


def main():
    log.info("=== Agent 3: Evolutionary Router ===")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    results = load_json(RESULTS_PATH)
    memory = load_json(MEMORY_PATH)

    if not results:
        log.error(f"No benchmark results found at {RESULTS_PATH}. Exiting.")
        sys.exit(2)

    # Extract CDLE metrics
    cdle = results.get("cdle", {})
    new_val_loss: float = cdle.get("val_loss", float("inf"))
    new_train_time: float = cdle.get("train_time_s", float("inf"))
    new_lpw: float = cdle.get("loss_per_watt", float("inf"))
    new_param_count: int = cdle.get("param_count", 0)

    # Current best
    best_lpw = memory.get("best_loss_per_watt")
    best_val = memory.get("best_val_loss")

    improvement_threshold: float = (
        cfg.get("evolution", {}).get("improvement_threshold", 0.01)
    )
    generation: int = memory.get("generation", 1)

    log.info(
        f"Generation {generation}: new_lpw={new_lpw:.6f}, "
        f"best_lpw={best_lpw if best_lpw else 'N/A'}"
    )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    is_first_run = best_lpw is None
    improved = (
        is_first_run
        or (new_lpw < best_lpw * (1.0 - improvement_threshold))
    )

    if improved:
        verdict = "merged"
        log.info(
            f"✅ MERGED — new model improves loss_per_watt by "
            + (
                f"{(best_lpw - new_lpw) / best_lpw * 100:.1f}%"
                if not is_first_run
                else "N/A (first run)"
            )
        )
        # Update best metrics
        memory["best_loss_per_watt"] = new_lpw
        memory["best_val_loss"] = new_val_loss
        # Snapshot the winning config
        memory["active_config"] = cfg.get("model", {})
    else:
        verdict = "discarded"
        log.info(
            f"❌ DISCARDED — new model (lpw={new_lpw:.6f}) does NOT improve "
            f"over best (lpw={best_lpw:.6f}). Reverting config."
        )
        # Restore best config if we have one
        if memory.get("active_config"):
            cfg["model"] = memory["active_config"]
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            log.info("Reverted config.yaml to best known configuration.")

    # ------------------------------------------------------------------
    # Update memory
    # ------------------------------------------------------------------
    # Find the proposed entry for this generation and update its verdict
    history = memory.get("history", [])
    updated = False
    for entry in reversed(history):
        if entry.get("generation") == generation and entry.get("status") == "proposed":
            entry["status"] = "evaluated"
            entry["verdict"] = verdict
            entry["val_loss"] = new_val_loss
            entry["train_time_s"] = new_train_time
            entry["loss_per_watt"] = new_lpw
            entry["param_count"] = new_param_count
            updated = True
            break

    if not updated:
        # Append a new record if no proposed entry exists
        history.append({
            "generation": generation,
            "status": "evaluated",
            "verdict": verdict,
            "val_loss": new_val_loss,
            "train_time_s": new_train_time,
            "loss_per_watt": new_lpw,
            "param_count": new_param_count,
            "config_snapshot": cfg.get("model", {}),
        })

    memory["history"] = history

    # Prune old history
    max_history = cfg.get("evolution", {}).get("max_history", 20)
    if len(memory["history"]) > max_history:
        memory["history"] = memory["history"][-max_history:]

    save_json(memory, MEMORY_PATH)

    # Print human-readable summary
    log.info("=== Agent 3 Summary ===")
    log.info(f"  Generation:       {generation}")
    log.info(f"  Verdict:          {verdict.upper()}")
    log.info(f"  New val_loss:     {new_val_loss:.4f}")
    log.info(f"  New lpw:          {new_lpw:.6f}")
    log.info(f"  Best lpw ever:    {memory['best_loss_per_watt']:.6f}")
    log.info("=== Agent 3 complete. ===")

    # Exit code: 0 = merged, 1 = discarded
    sys.exit(0 if improved else 1)


if __name__ == "__main__":
    main()
