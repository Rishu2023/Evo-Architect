"""
agents/agent1_physicist.py
===========================
Agent 1: The Theoretical Physicist

Role:
  Read the evolutionary memory (past benchmark results) and use the GitHub
  Models free API to generate an improved `config.yaml` for the next
  generation of the CDLE architecture.

API:
  - Endpoint: https://models.inference.ai.azure.com  (OpenAI-compatible)
  - Model:    gpt-4o-mini
  - Auth:     GITHUB_TOKEN environment variable (automatically available in
              GitHub Actions — no extra secrets required, completely free)

Workflow:
  1. Load evolutionary_memory.json (create empty one if missing).
  2. Build a structured prompt describing the current best architecture,
     past failures, and the goal (minimise loss_per_watt).
  3. Call gpt-4o-mini to propose a new set of hyperparameters.
  4. Parse the LLM response as YAML.
  5. Write the new config.yaml to the working directory.
  6. Update evolutionary_memory.json with the proposed generation metadata.

The script is designed to be called from GitHub Actions but also works
locally (set GITHUB_TOKEN in your environment).
"""

import os
import sys
import json
import logging
import re
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MEMORY_PATH = "evolutionary_memory.json"
CONFIG_PATH = "config.yaml"

# GitHub Models endpoint (OpenAI-compatible)
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_MODELS_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def load_memory(path: str = MEMORY_PATH) -> dict:
    """Load the evolutionary memory JSON, creating an empty one if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Initialise empty memory
    return {
        "generation": 0,
        "best_loss_per_watt": None,
        "best_val_loss": None,
        "history": [],
        "active_config": None,
    }


def save_memory(memory: dict, path: str = MEMORY_PATH) -> None:
    """Persist the evolutionary memory to disk."""
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)
    log.info(f"Saved evolutionary memory → {path}")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_physicist_prompt(memory: dict, current_config: dict) -> str:
    """
    Build the structured prompt for gpt-4o-mini.

    The prompt describes:
      - Current best metrics (if any).
      - Last few generations and their verdicts.
      - Constraints (CPU-only, ≤8 M params, ≤8 min training).
      - Current config.yaml values.
      - Exact output format required (YAML block).

    Args:
        memory:         Evolutionary memory dict.
        current_config: Current config.yaml as a dict.

    Returns:
        Prompt string.
    """
    generation = memory.get("generation", 0) + 1
    best_lpw = memory.get("best_loss_per_watt")
    best_val = memory.get("best_val_loss")

    # Summarise history (last 5 entries)
    history = memory.get("history", [])
    recent = history[-5:] if history else []
    history_text = json.dumps(recent, indent=2) if recent else "No history yet."

    model_cfg = current_config.get("model", {})
    train_cfg = current_config.get("training", {})

    prompt = f"""You are a theoretical physicist and AI architect specialising in efficient neural architectures.

TASK: Propose improved hyperparameters for generation {generation} of the CDLE (Continuous Dynamic Liquid Engine) language model.

OBJECTIVE: Minimise `loss_per_watt = val_loss / train_time_s` on the TinyStories dataset.
Lower is better. The model must train to convergence in under 8 minutes on a CPU-only GitHub Actions runner.

CONSTRAINTS (non-negotiable):
- Total parameters: 1 M to 8 M (no more, no less)
- Training time: under 480 seconds (8 minutes) on ubuntu-latest CPU
- vocab_size must stay at 256 (byte-level, no tokeniser)
- seq_len must stay at 256
- d_model must be divisible by 8

CURRENT BEST METRICS:
- best_loss_per_watt: {best_lpw if best_lpw is not None else "N/A (first run)"}
- best_val_loss:      {best_val if best_val is not None else "N/A (first run)"}

RECENT GENERATION HISTORY (last 5 runs):
{history_text}

CURRENT CONFIG (model section):
{json.dumps(model_cfg, indent=2)}

CURRENT CONFIG (training section):
{json.dumps(train_cfg, indent=2)}

PHYSICS INTUITION TO APPLY:
- More d_state increases SSM memory but costs more FLOPs (linear in d_state).
- More n_layers increases capacity but also training time.
- Larger d_model improves representation but grows params quadratically.
- LTC tau_base: larger = slower dynamics (better for long-range patterns).
- Higher learning_rate can speed convergence but risks instability.
- Forward-Forward ff_threshold: tune so ~50% of activations are "good".

OUTPUT FORMAT (reply with ONLY this YAML block, no prose, no markdown fences):
model:
  d_model: <integer>
  n_layers: <integer>
  d_state: <integer>
  d_ff: <integer>
  vocab_size: 256
  seq_len: 256
  ltc_tau_base: <float>
  ff_threshold: <float>
  dropout: <float>
training:
  batch_size: <integer>
  max_steps: <integer>
  val_every: 100
  val_steps: 20
  learning_rate: <float>
  weight_decay: <float>
  grad_clip: <float>
  warmup_steps: <integer>
data:
  dataset_name: "roneneldan/TinyStories"
  num_train_samples: 9000
  num_val_samples: 1000
  cache_path: "data/tinystories_cache.pt"
evolution:
  memory_path: "evolutionary_memory.json"
  results_path: "benchmark_results.json"
  improvement_threshold: 0.01
  max_history: 20
"""
    return prompt


# ---------------------------------------------------------------------------
# GitHub Models API call
# ---------------------------------------------------------------------------

def call_github_models(prompt: str) -> str:
    """
    Call the GitHub Models API (gpt-4o-mini) with the given prompt.

    Requires GITHUB_TOKEN environment variable (set automatically in Actions).

    Args:
        prompt: Full system+user prompt.

    Returns:
        Raw text response from the model.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN environment variable not set. "
            "This should be automatically available in GitHub Actions."
        )

    # Import here so the module can be imported without openai installed
    from openai import OpenAI

    client = OpenAI(
        base_url=GITHUB_MODELS_ENDPOINT,
        api_key=token,
    )

    log.info(f"Calling GitHub Models API ({GITHUB_MODELS_MODEL})…")
    response = client.chat.completions.create(
        model=GITHUB_MODELS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI architect. Respond ONLY with valid YAML. "
                    "No markdown code fences. No prose. Just the YAML."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    text = response.choices[0].message.content.strip()
    log.info("Received response from GitHub Models API.")
    return text


# ---------------------------------------------------------------------------
# YAML parsing + validation
# ---------------------------------------------------------------------------

def parse_and_validate_config(raw_yaml: str, current_config: dict) -> dict:
    """
    Parse the LLM's YAML response and validate / clamp all values.

    Falls back gracefully to the current config if parsing fails.

    Args:
        raw_yaml:       Raw string from the LLM.
        current_config: Current config dict (used as fallback).

    Returns:
        Validated config dict.
    """
    # Strip markdown fences if the model added them despite instructions
    raw_yaml = re.sub(r"```[a-z]*\n?", "", raw_yaml).strip()

    try:
        new_cfg = yaml.safe_load(raw_yaml)
        if not isinstance(new_cfg, dict):
            raise ValueError("Parsed YAML is not a dict")
    except Exception as e:
        log.warning(f"YAML parse failed ({e}). Using current config with small mutation.")
        new_cfg = current_config.copy()

    # Ensure required sections exist
    if "model" not in new_cfg:
        new_cfg["model"] = current_config.get("model", {}).copy()
    if "training" not in new_cfg:
        new_cfg["training"] = current_config.get("training", {}).copy()
    if "data" not in new_cfg:
        new_cfg["data"] = current_config.get("data", {}).copy()
    if "evolution" not in new_cfg:
        new_cfg["evolution"] = current_config.get("evolution", {}).copy()

    m = new_cfg["model"]
    t = new_cfg["training"]

    # Hard constraints: keep fixed values
    m["vocab_size"] = 256
    m["seq_len"] = 256

    # Clamp individual hyperparameters first
    # Apply divisibility by 8 FIRST, then clamp, so both constraints hold
    d_model_raw = int(m.get("d_model", 128))
    d_model_raw = max(8, round(d_model_raw / 8) * 8)
    m["d_model"] = max(64, min(256, d_model_raw))

    m["n_layers"] = max(2, min(8, int(m.get("n_layers", 4))))
    m["d_state"] = max(8, min(64, int(m.get("d_state", 16))))
    m["d_ff"] = max(64, min(512, int(m.get("d_ff", 256))))
    m["ltc_tau_base"] = max(0.1, min(5.0, float(m.get("ltc_tau_base", 1.0))))
    m["ff_threshold"] = max(0.5, min(10.0, float(m.get("ff_threshold", 2.0))))
    m["dropout"] = max(0.0, min(0.3, float(m.get("dropout", 0.0))))

    # Validate actual parameter count using a formula-based estimate (no torch).
    # If outside [1M, 8M], scale d_model up/down by one step (8 params) until satisfied.
    # Max iterations: d_model range is 64–256 (192 values), step size 8 → at most 24 steps;
    # 32 gives a safe margin.
    MIN_PARAMS = 1_000_000
    MAX_PARAMS = 8_000_000
    # Compute initial estimate before the loop so `estimated` is always defined.
    estimated = _estimate_cdle_params(
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        d_state=m["d_state"],
        vocab_size=m["vocab_size"],
        seq_len=m["seq_len"],
    )
    for _ in range(32):
        if estimated < MIN_PARAMS and m["d_model"] < 256:
            m["d_model"] = min(256, m["d_model"] + 8)
        elif estimated > MAX_PARAMS and m["d_model"] > 64:
            m["d_model"] = max(64, m["d_model"] - 8)
        else:
            break
        estimated = _estimate_cdle_params(
            d_model=m["d_model"],
            n_layers=m["n_layers"],
            d_state=m["d_state"],
            vocab_size=m["vocab_size"],
            seq_len=m["seq_len"],
        )
    log.info(f"Estimated CDLE param count: {estimated:,} (target 1M–8M)")

    # Clamp training hyperparameters
    t["batch_size"] = max(8, min(128, int(t.get("batch_size", 32))))
    t["max_steps"] = max(100, min(600, int(t.get("max_steps", 500))))
    t["learning_rate"] = max(1e-5, min(1e-2, float(t.get("learning_rate", 3e-4))))
    t["weight_decay"] = max(0.0, min(0.1, float(t.get("weight_decay", 1e-2))))
    t["grad_clip"] = max(0.1, min(5.0, float(t.get("grad_clip", 1.0))))
    t["warmup_steps"] = max(0, min(200, int(t.get("warmup_steps", 50))))
    t["val_every"] = 100
    t["val_steps"] = 20

    return new_cfg


def _estimate_cdle_params(
    d_model: int,
    n_layers: int,
    d_state: int,
    vocab_size: int = 256,
    seq_len: int = 256,
) -> int:
    """
    Lightweight formula-based estimate of CDLEModel parameter count.

    No torch dependency — safe to call during config validation.

    Breakdown per CDLEBlock:
      SelectiveSSM: in_proj(D→2D) + x_proj(D→2N+1) + A_log(N) + D_skip(D) + out_proj(D→D)
      LTC:          3 × Linear(D→D, bias) + LayerNorm(D)
      FFLayer:      Linear(D→D, bias) + LayerNorm(D)
      2 × pre-norm: LayerNorm(D)

    Args:
        d_model:    Hidden dimension.
        n_layers:   Number of CDLE blocks.
        d_state:    SSM state size.
        vocab_size: Token vocabulary size.
        seq_len:    Maximum sequence length (for positional embeddings).

    Returns:
        Estimated parameter count (int).
    """
    D, N = d_model, d_state

    # Embeddings (tied lm_head counts once)
    embedding_params = vocab_size * D + seq_len * D

    # Per CDLEBlock
    ssm_params = (
        D * (2 * D)           # in_proj
        + D * (2 * N + 1)     # x_proj
        + N                   # A_log
        + D                   # D_skip
        + D * D               # out_proj
    )
    ltc_params = 3 * (D * D + D) + 2 * D   # 3 linears + LayerNorm
    ff_params = (D * D + D) + 2 * D        # linear + LayerNorm
    pre_norms = 2 * 2 * D                  # 2 × LayerNorm (weight + bias)
    per_block = ssm_params + ltc_params + ff_params + pre_norms

    # Final LayerNorm
    final_norm = 2 * D

    return embedding_params + n_layers * per_block + final_norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Agent 1: Theoretical Physicist ===")

    # Load current state
    memory = load_memory(MEMORY_PATH)
    with open(CONFIG_PATH) as f:
        current_config = yaml.safe_load(f)

    generation = memory.get("generation", 0) + 1
    log.info(f"Proposing config for generation {generation}.")

    # Build prompt
    prompt = build_physicist_prompt(memory, current_config)
    log.info(f"Prompt length: {len(prompt)} characters")

    # Call LLM
    raw_response = call_github_models(prompt)
    log.info(f"Raw LLM response:\n{raw_response}")

    # Parse + validate
    new_config = parse_and_validate_config(raw_response, current_config)
    log.info(f"Validated new config:\n{yaml.dump(new_config, default_flow_style=False)}")

    # Save new config
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    log.info(f"Wrote new config to {CONFIG_PATH}")

    # Update memory with proposal record
    memory["generation"] = generation
    memory["history"].append({
        "generation": generation,
        "status": "proposed",
        "config_snapshot": new_config.get("model", {}),
        "notes": "Proposed by Agent 1 (Theoretical Physicist)",
    })

    # Prune history
    max_history = memory.get("max_history", 20) or current_config.get(
        "evolution", {}
    ).get("max_history", 20)
    if len(memory["history"]) > max_history:
        memory["history"] = memory["history"][-max_history:]

    save_memory(memory, MEMORY_PATH)
    log.info(f"=== Agent 1 complete. Generation {generation} proposed. ===")


if __name__ == "__main__":
    main()
