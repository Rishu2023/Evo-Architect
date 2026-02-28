# Evo-Architect 🧬

A **Continuous Dynamic Liquid Engine (CDLE)** — an automated, self-evolving AI research laboratory that runs 24/7 on **100% free** GitHub infrastructure.

---

## 🏗️ Architecture Overview

The system implements a **four-agent evolutionary loop** that continuously proposes, benchmarks, judges, and verifies new neural architectures:

```
┌─────────────────────────────────────────────────────────────────┐
│                 EVO-ARCHITECT: 24/7 EVOLUTION LOOP              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Agent 1     │───▶│  Agent 2     │───▶│  Agent 3 + 4     │  │
│  │  Physicist   │    │  Benchmarker │    │  Judge+Verifier  │  │
│  │  (LLM Ideas) │    │  (Train+Eval)│    │  (Merge/Delete)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         ▲                                         │             │
│         └──────────── evolutionary_memory.json ◀──┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Core Innovation: CDLE (Continuous Dynamic Liquid Engine)

The **CDLE** model combines four cutting-edge ideas into one architecture:

| Component | Description |
|-----------|-------------|
| **Byte-level Input** | No BPE tokenizer — raw character/byte embeddings for true universal learning |
| **Mamba-style SSM** | Selective State Space Model for O(L) sequence modeling instead of O(L²) attention |
| **Liquid Dynamics** | Liquid Time-Constant (LTC) layers that dynamically adjust based on input complexity |
| **Forward-Forward Learning** | Localized Hebbian-style updates during forward pass — no full backprop required |

---

## 📁 Directory Structure

```
evo-architect/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Hyperparameters
├── .gitignore
├── models/
│   ├── cdle_base.py                   # Core CDLE architecture
│   └── baseline_transformer.py        # Tiny GPT-2 baseline for comparison
├── agents/
│   ├── agent1_physicist.py            # Theoretical Physicist (LLM via GitHub Models)
│   ├── agent2_benchmark.py            # Train + benchmark both models
│   ├── agent3_evolutionary_router.py  # Compare results, merge or delete
│   └── agent4_verifier.py             # SymPy gradient stability + proof
├── data/
│   └── prepare_tinystories.py         # Download & prep first 10k TinyStories samples
├── utils/
│   └── metrics.py                     # Shared metrics utilities
└── .github/
    └── workflows/
        ├── generate-architecture.yml  # Agent 1: hourly + manual trigger
        ├── benchmark.yml              # Agent 2: runs on test-branch push
        └── judge-and-merge.yml        # Agents 3+4: post-benchmark evaluation
```

---

## 🚀 Quick Start

### Prerequisites

- A GitHub account (free tier is sufficient)
- Fork or clone this repository

### 1. Enable GitHub Actions

1. Go to your fork → **Settings** → **Actions** → **General**
2. Set **Workflow permissions** to **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**

### 2. Install Dependencies Locally (Optional)

```bash
pip install -r requirements.txt
```

### 3. Start the Evolution Loop

1. Go to **Actions** tab in your repository
2. Click **"Generate New Architecture"** workflow
3. Click **"Run workflow"** → **Run workflow**

This triggers Agent 1 which proposes a new architecture, commits it to `test-branch`, which automatically triggers Agent 2 (benchmarking), then Agent 3+4 (judging+verifying).

### 4. View Results

Results are stored in `evolutionary_memory.json` (committed back to the repo after each run). Check the Actions tab logs for detailed output.

---

## 📊 Results Dashboard (Architecture)

The system tracks these metrics per architecture generation:

| Metric | Description |
|--------|-------------|
| `val_loss` | Validation loss on TinyStories |
| `train_time_s` | Total training time in seconds |
| `loss_per_watt` | `val_loss / train_time_s` (efficiency proxy) |
| `flops_estimate` | Estimated FLOPs for one forward pass |
| `param_count` | Number of trainable parameters |
| `stability_score` | SymPy-verified gradient stability (0–1) |
| `generation` | Evolution generation number |

### Sample `evolutionary_memory.json` structure:

```json
{
  "generation": 3,
  "best_loss_per_watt": 0.0023,
  "history": [
    {
      "generation": 1,
      "model": "cdle_v1",
      "val_loss": 2.34,
      "train_time_s": 180,
      "loss_per_watt": 0.013,
      "verdict": "merged",
      "notes": "Initial baseline architecture"
    }
  ],
  "active_config": { "d_model": 128, "n_layers": 4 }
}
```

---

## ⚙️ Configuration (`config.yaml`)

All hyperparameters are in `config.yaml`:

```yaml
model:
  d_model: 128        # Embedding dimension (1M–8M params target)
  n_layers: 4         # Number of CDLE layers
  d_state: 16         # SSM state dimension

training:
  batch_size: 32
  max_steps: 500      # ~5–7 min on CPU
  learning_rate: 3e-4
```

---

## 🤖 Agent Details

### Agent 1: Theoretical Physicist
- **Trigger**: Hourly schedule OR manual `workflow_dispatch`
- **Tool**: GitHub Models free API (gpt-4o-mini via `models.inference.ai.azure.com`)
- **Job**: Read `evolutionary_memory.json`, propose new hyperparameter mutations or architectural changes, write updated `config.yaml` to `test-branch`

### Agent 2: Benchmarker
- **Trigger**: Push to `test-branch`
- **Job**: Train CDLE + baseline transformer on TinyStories (10k samples, ≤8 min), compute `loss_per_watt`, save results to `benchmark_results.json`

### Agent 3: Evolutionary Router
- **Trigger**: After Agent 2 completes
- **Job**: Compare new results vs. evolutionary memory, decide merge (better) or delete (worse), update `evolutionary_memory.json`

### Agent 4: Verifier
- **Trigger**: Same workflow as Agent 3
- **Job**: Use SymPy to analytically verify gradient stability of proposed architecture, append proof to results

---

## 💸 Cost: $0.00

| Resource | Cost |
|----------|------|
| GitHub Actions (ubuntu-latest, CPU) | Free (≤2000 min/month) |
| GitHub Models API (gpt-4o-mini) | Free with GITHUB_TOKEN |
| TinyStories dataset (Hugging Face) | Free |
| Repository storage | Free (public repo) |

---

## 🔬 Technical Deep Dive

### CDLE Architecture

```
Input (bytes/chars)
      │
      ▼
[Byte Embedding Layer]  ← d_vocab=256, d_model=128
      │
      ▼
[CDLE Block] × n_layers
  ├── [Selective SSM]       ← Mamba-style: O(L) complexity
  ├── [Liquid Time-Constant]  ← LTC: dynamic routing
  └── [Forward-Forward Update]  ← Localized learning
      │
      ▼
[Output Head]  → next-byte logits
```

### Forward-Forward Learning

Instead of standard backpropagation, each layer learns to maximize "goodness" on real data and minimize it on synthetic negative samples:

```
goodness(h) = sum(h²)
layer_loss = log(1 + exp(-(goodness(h_pos) - threshold)))
           + log(1 + exp(goodness(h_neg) - threshold))
```

---

## 🔄 24/7 Operation

The system runs continuously via GitHub Actions scheduled triggers:

1. **Every hour**: Agent 1 proposes a new architecture variant
2. **Automatically**: Agents 2, 3, 4 evaluate and integrate the best variants
3. **Self-improving**: Each generation builds on the best previous architecture
4. **Memory**: `evolutionary_memory.json` tracks all history for Agent 1 to learn from

---

## 📜 License

MIT License — completely free to use, modify, and deploy.
