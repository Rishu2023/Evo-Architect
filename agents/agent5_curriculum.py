"""
agents/agent5_curriculum.py
============================
Agent 5: Curriculum Evolution Agent

Role:
  Propose and evolve training curricula for the CDLE model.  Instead of
  training on the full dataset from step 0, the curriculum agent adjusts
  *difficulty* over time — starting with shorter/simpler samples and
  progressively increasing length and complexity.

Algorithm:
  1. Load current curriculum state (difficulty, ordering strategy).
  2. Evaluate recent benchmark results to decide whether to:
     a) Increase difficulty (model is mastering current level).
     b) Decrease difficulty (model is struggling → regression).
     c) Hold steady (inconclusive signal).
  3. Write updated curriculum state to archive/curriculum_state.json.
  4. Optionally propose a new ordering of the training data for Agent 2.

This is called from the curriculum-evolve.yml workflow.
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
RESULTS_PATH = "benchmark_results.json"
MEMORY_PATH = "evolutionary_memory.json"


def load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str) -> None:
    """Save a dict as a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved → {path}")


def compute_difficulty_signal(results: dict, memory: dict) -> str:
    """
    Determine whether to increase, decrease, or hold curriculum difficulty.

    Strategy:
      - If recent val_loss improved by >5% over previous generation → increase.
      - If recent val_loss degraded by >10% → decrease.
      - Otherwise → hold.

    Args:
        results: Latest benchmark_results.json content.
        memory:  Evolutionary memory dict.

    Returns:
        One of "increase", "decrease", "hold".
    """
    cdle = results.get("cdle", {})
    current_loss = cdle.get("val_loss", float("inf"))
    best_loss = memory.get("best_val_loss")

    if best_loss is None or best_loss <= 0:
        return "hold"

    improvement = (best_loss - current_loss) / best_loss

    if improvement > 0.05:
        return "increase"
    elif improvement < -0.10:
        return "decrease"
    return "hold"


def evolve_curriculum(cfg: dict, signal: str, state: dict) -> dict:
    """
    Update the curriculum state based on the difficulty signal.

    Args:
        cfg:    Config dict (curriculum section).
        signal: "increase", "decrease", or "hold".
        state:  Current curriculum state dict.

    Returns:
        Updated curriculum state dict.
    """
    cur_cfg = cfg.get("curriculum", {})
    step = cur_cfg.get("difficulty_step", 0.1)
    max_diff = cur_cfg.get("max_difficulty", 1.0)
    min_diff = cur_cfg.get("initial_difficulty", 0.3)

    current_difficulty = state.get("difficulty", min_diff)
    generation = state.get("generation", 0) + 1

    if signal == "increase":
        new_difficulty = min(max_diff, current_difficulty + step)
        log.info(f"📈 Increasing difficulty: {current_difficulty:.2f} → {new_difficulty:.2f}")
    elif signal == "decrease":
        new_difficulty = max(min_diff, current_difficulty - step)
        log.info(f"📉 Decreasing difficulty: {current_difficulty:.2f} → {new_difficulty:.2f}")
    else:
        new_difficulty = current_difficulty
        log.info(f"➡️ Holding difficulty at {current_difficulty:.2f}")

    # Determine data ordering strategy based on difficulty
    if new_difficulty < 0.4:
        ordering = "short_first"
        max_seq_fraction = 0.5
    elif new_difficulty < 0.7:
        ordering = "mixed"
        max_seq_fraction = 0.75
    else:
        ordering = "full"
        max_seq_fraction = 1.0

    history = state.get("history", [])
    history.append({
        "generation": generation,
        "signal": signal,
        "difficulty": round(new_difficulty, 3),
        "ordering": ordering,
    })

    # Keep last 20 entries
    if len(history) > 20:
        history = history[-20:]

    return {
        "difficulty": round(new_difficulty, 3),
        "ordering": ordering,
        "max_seq_fraction": max_seq_fraction,
        "generation": generation,
        "history": history,
    }


def main():
    log.info("=== Agent 5: Curriculum Evolution ===")

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Load current state
    curriculum_path = cfg.get("curriculum", {}).get(
        "curriculum_path", "archive/curriculum_state.json"
    )
    state = load_json(curriculum_path)
    results = load_json(RESULTS_PATH)
    memory = load_json(MEMORY_PATH)

    # Compute difficulty signal
    signal = compute_difficulty_signal(results, memory)
    log.info(f"Difficulty signal: {signal}")

    # Evolve curriculum
    new_state = evolve_curriculum(cfg, signal, state)

    # Save updated state
    save_json(new_state, curriculum_path)

    # Summary
    log.info("=== Agent 5 Summary ===")
    log.info(f"  Difficulty:  {new_state['difficulty']}")
    log.info(f"  Ordering:    {new_state['ordering']}")
    log.info(f"  Max seq %:   {new_state['max_seq_fraction']}")
    log.info(f"  Generation:  {new_state['generation']}")
    log.info("=== Agent 5 complete. ===")


if __name__ == "__main__":
    main()
