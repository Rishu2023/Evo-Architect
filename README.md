# Evo-Architect 🧬

A **self-evolving AI research laboratory** powered by the **Continuous Dynamic Liquid Engine (CDLE v2)** — running 24/7 on **100% free** GitHub infrastructure. Zero cloud bills, zero human babysitting.

> **Why public?** GitHub gives public repositories **unlimited** Actions minutes. This repo exploits that to run a perpetual, multi-agent neural-architecture search — for $0.00 forever.

---

## 🏗️ Architecture Overview

The system implements a **six-agent evolutionary loop** that continuously proposes, benchmarks, routes, verifies, curricularises, and ingests literature for new neural architectures:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     EVO-ARCHITECT: 24/7 EVOLUTION LOOP                       │
│                                                                              │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │  Agent 1      │──▶│  Agent 2      │──▶│  Agent 3      │                  │
│  │  Theoretical  │   │  Benchmarker  │   │  Evolutionary  │                 │
│  │  Physicist    │   │  (Multi-task  │   │  Router        │                 │
│  │  (LLM Ideas) │   │   + Pareto)   │   │  (QD MAP-Elites│                 │
│  └───────────────┘   └───────────────┘   │   Archive)    │                  │
│         ▲                                 └───────┬───────┘                  │
│         │                                         │                          │
│         │                                         ▼                          │
│         │                                 ┌───────────────┐                  │
│         │                                 │  Agent 4      │                  │
│         │                                 │  Formal       │                  │
│         │                                 │  Verifier     │                  │
│         │                                 │  (SymPy +     │                  │
│         │                                 │   Stress)     │                  │
│         │                                 └───────┬───────┘                  │
│         │                                         │                          │
│         │         ┌───────────────┐                │                          │
│         │         │  Agent 5      │◀───────────────┘                          │
│         │         │  Curriculum   │                                           │
│         │         │  Evolution    │                                           │
│         │         └───────┬───────┘                                           │
│         │                 │                                                   │
│         │                 ▼                                                   │
│         │  evolutionary_memory.json + archive/qd_population.json             │
│         └────────────────────────────────────────────────────────             │
│                                                                              │
│  ┌─────────────────────┐  (weekly, async)                                    │
│  │  Literature Agent   │──▶ evolutionary_memory.json["literature"]           │
│  │  (arXiv / Semantic  │                                                     │
│  │   Scholar ingest)   │                                                     │
│  └─────────────────────┘                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Innovation — CDLE v2

The **Continuous Dynamic Liquid Engine v2** (`models/cdle_base.py`) fuses eight ideas into a single, CPU-friendly architecture (1 M–12 M params):

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Byte-level continuous multimodal streams** | No BPE tokenizer — raw byte embeddings (`vocab_size=256`) for truly universal input (text, code, binary) |
| 2 | **Mamba-style Selective SSM** | O(L) sequence modelling via selective state spaces instead of O(L²) attention |
| 3 | **Fractal / Hierarchical SSM state** | Multi-scale temporal modelling with `fractal_levels` recursive state layers |
| 4 | **Event-driven sparse Liquid routing** | Liquid Time-Constant (LTC) neurons with a **complexity gate** — simple inputs skip heavy compute (`complexity_gate_threshold` in `config.yaml`) |
| 5 | **Configurable Dynamic Forward-Forward** | Two FF variants selectable via `ff_variant`: **Distance-FF** (distance-based goodness) and **Self-Contrastive** (contrastive within-layer objective) — no full backprop required |
| 6 | **Energy proxy** | Combined FLOPs/sec + linear watt estimate (`cpu_tdp_watts × utilisation`) for cost-aware architecture search |
| 7 | **Liquid Time-Constant dynamics** | Input-adaptive gating with configurable `ltc_tau_base` |
| 8 | **Pure PyTorch 2.x** | No custom CUDA kernels — runs on free GitHub Actions CPU runners |

---

## 📁 Directory Structure

```
evo-architect/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (torch, sympy, requests, …)
├── config.yaml                            # All hyperparameters — Agent 1 mutates this
├── benchmark_results.json                 # Latest benchmark output
├── .gitignore
│
├── models/
│   ├── __init__.py
│   ├── cdle_base.py                       # CDLE v2 core (SSM + fractal + LTC + FF)
│   └── baseline_transformer.py            # Tiny GPT-2 baseline for comparison
│
├── agents/
│   ├── __init__.py
│   ├── agent1_physicist.py                # Agent 1 — Theoretical Physicist (LLM proposals)
│   ├── agent2_benchmark.py                # Agent 2 — Train + multi-task benchmark
│   ├── agent3_evolutionary_router.py      # Agent 3 — QD MAP-Elites evolutionary router
│   ├── agent4_verifier.py                 # Agent 4 — SymPy proofs + stress tests
│   ├── agent5_curriculum.py               # Agent 5 — Curriculum evolution
│   └── literature_ingest.py               # Literature Agent — weekly arXiv ingestion
│
├── benchmarks/
│   ├── __init__.py
│   └── multi_task_eval.py                 # Multi-task suite (TinyStories, PIQA, GSM8K, ARC)
│
├── archive/
│   ├── qd_population.json                 # QD MAP-Elites archive (4×3 niches)
│   └── curriculum_state.json              # Current curriculum difficulty state
│
├── data/
│   ├── prepare_tinystories.py             # Download & prep TinyStories samples
│   └── prepare_datasets.py               # Multi-task data preparation
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                         # Shared metrics & Pareto utilities
│
├── docs/
│   ├── _config.yml                        # Jekyll / GitHub Pages config
│   └── index.md                           # Live dashboard (HTML + JS)
│
└── .github/
    ├── ISSUE_TEMPLATE/
    │   └── steer-evolution.md             # Issue template for steering evolution
    └── workflows/
        ├── generate-architecture.yml      # Agent 1 — hourly + manual trigger
        ├── benchmark.yml                  # Agent 2 — on test-branch push
        ├── judge-and-merge.yml            # Agents 3 + 4 — post-benchmark
        ├── curriculum-evolve.yml          # Agent 5 — after judge-and-merge
        ├── deploy-dashboard.yml           # GitHub Pages — on push to main
        └── literature-weekly.yml          # Literature Agent — Sundays 06:00 UTC
```

---

## 🚀 Quick Start

### 1. Fork the Repository

Click **Fork** in the top-right corner. The repo must remain **public** for unlimited free Actions minutes.

### 2. Enable GitHub Actions

1. Go to your fork → **Settings** → **Actions** → **General**
2. Set **Workflow permissions** to **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**

### 3. (Optional) Install Locally

```bash
pip install -r requirements.txt
```

### 4. Start the Evolution Loop

1. Go to the **Actions** tab
2. Click **"Generate New Architecture"**
3. Click **"Run workflow"** → **Run workflow**

Agent 1 proposes a new architecture → commits to `test-branch` → triggers Agent 2 (multi-task benchmark) → triggers Agents 3+4 (routing + verification) → triggers Agent 5 (curriculum update). The literature agent runs every Sunday automatically.

### 5. Watch It Evolve

- **Actions tab** — real-time workflow logs
- **`evolutionary_memory.json`** — full history committed to the repo
- **Dashboard** — live charts at your GitHub Pages URL (see below)

---

## 📊 Dashboard

A live GitHub Pages dashboard is deployed automatically on every push to `main`:

> **URL**: `https://<your-username>.github.io/Evo-Architect/`

The dashboard (`docs/index.md`) shows:

| Section | What It Displays |
|---------|------------------|
| **Current Status** | Generation counter, best loss/watt, best val loss, QD archive coverage % |
| **Evolution Leaderboard** | Last 10 generations with verdict, val loss, loss/watt, params, stability |
| **Evolution Plot** | Line chart of loss/watt over generations |
| **QD Archive Grid** | 4×3 heatmap of complexity × sparsity niches (MAP-Elites) |
| **Model Playground** | Mock byte-level text generation demo |

Data is read from `evolutionary_memory.json` and `qd_population.json` at page load.

---

## 🎯 Steering via Issues

You can guide the evolution without touching code. Open an issue using the built-in template:

**Actions** → **New Issue** → **🎯 Steer Evolution** (or use `.github/ISSUE_TEMPLATE/steer-evolution.md`)

The template lets you:

- ✅ Request architecture changes (e.g., increase `d_model`, switch `ff_variant`)
- ✅ Add new objectives (e.g., prioritise sparsity)
- ✅ Modify training (e.g., curriculum schedule, learning rate)
- ✅ Propose new benchmark tasks
- ✅ Set priority (high / medium / low)
- ✅ Attach suggested hyperparameters as YAML

Agent 1 reads open steering issues and incorporates them into its next architecture proposal.

---

## 📈 Metrics & Scoring

### Pareto Multi-Objective Scoring

Every candidate architecture is evaluated on **five objectives** simultaneously:

| Objective | Source | Direction |
|-----------|--------|-----------|
| **Loss / second** | `val_loss / train_time_s` | ↓ Lower is better |
| **Sparsity** | Fraction of complexity-gated activations skipped | ↑ Higher is better |
| **Memory** | Peak parameter count vs `max_params` cap (12 M) | ↓ Lower is better |
| **Generalisation** | Multi-task average across PIQA, GSM8K, ARC (`benchmarks/multi_task_eval.py`) | ↑ Higher is better |
| **Continual adaptation** | Curriculum difficulty progression rate (`archive/curriculum_state.json`) | ↑ Higher is better |

Only **Pareto-dominant** or **niche-filling** candidates survive.

### QD MAP-Elites Archive

The evolutionary router (`agents/agent3_evolutionary_router.py`) maintains a **Quality-Diversity** archive in `archive/qd_population.json`:

```
           Sparsity →
           Low    Med    High
         ┌──────┬──────┬──────┐
  High   │ (3,0)│ (3,1)│ (3,2)│
         ├──────┼──────┼──────┤
Complexity Mid-H │ (2,0)│ (2,1)│ (2,2)│
    ↓     ├──────┼──────┼──────┤
  Mid-L  │ (1,0)│ (1,1)│ (1,2)│
         ├──────┼──────┼──────┤
  Low    │ (0,0)│ (0,1)│ (0,2)│
         └──────┴──────┴──────┘
```

- **4 complexity bins** × **3 sparsity bins** = **12 niches** (`qd_max_species: 12`)
- Each niche stores the **best-so-far** individual for that region of behaviour space
- Ensures diversity: simple-dense models coexist with complex-sparse ones

### Energy Proxy Scoring

Every benchmark run estimates energy cost (`config.yaml` → `energy` section):

```
energy_score = flops_weight × (FLOPs / second) + watt_weight × (cpu_tdp_watts × utilisation)
```

Defaults: `flops_weight: 0.5`, `watt_weight: 0.5`, `cpu_tdp_watts: 65.0`. Lower energy score = better.

---

## 🤖 Agent Details

### Agent 1: Theoretical Physicist — `agents/agent1_physicist.py`

| | |
|---|---|
| **Trigger** | Hourly cron schedule OR manual `workflow_dispatch` |
| **Workflow** | `.github/workflows/generate-architecture.yml` |
| **Tool** | GitHub Models free API (gpt-4o-mini via `models.inference.ai.azure.com`) |
| **Inputs** | `evolutionary_memory.json`, open steering issues, literature summaries |
| **Outputs** | Mutated `config.yaml` committed to `test-branch` |

Reads the full evolutionary history, recent literature, and any open `[STEER]` issues, then uses the LLM to propose hyperparameter mutations or architectural changes.

### Agent 2: Benchmarker — `agents/agent2_benchmark.py`

| | |
|---|---|
| **Trigger** | Push to `test-branch` |
| **Workflow** | `.github/workflows/benchmark.yml` |
| **Tasks** | Train CDLE + baseline on TinyStories (9 k samples), run multi-task eval (PIQA, GSM8K, ARC — 1 k each) |
| **Outputs** | `benchmark_results.json` with per-task loss/accuracy/time, Pareto scores, energy proxy |

Total runtime target: **≤12 minutes** on a free `ubuntu-latest` runner.

### Agent 3: Evolutionary Router — `agents/agent3_evolutionary_router.py`

| | |
|---|---|
| **Trigger** | After Agent 2 completes (`workflow_run`) |
| **Workflow** | `.github/workflows/judge-and-merge.yml` |
| **Algorithm** | QD MAP-Elites: place candidate in the correct (complexity, sparsity) niche; keep if it dominates the existing occupant |
| **Outputs** | Updated `evolutionary_memory.json`, updated `archive/qd_population.json` |

### Agent 4: Formal Verifier — `agents/agent4_verifier.py`

| | |
|---|---|
| **Trigger** | Same workflow as Agent 3 |
| **Workflow** | `.github/workflows/judge-and-merge.yml` |
| **Checks** | SymPy-based analytical gradient stability proofs + numerical stress tests (NaN detection, extreme inputs) |
| **Outputs** | `stability_score` (0–1) appended to benchmark results |

### Agent 5: Curriculum Evolution — `agents/agent5_curriculum.py`

| | |
|---|---|
| **Trigger** | After Agents 3+4 complete (`workflow_run`) OR manual |
| **Workflow** | `.github/workflows/curriculum-evolve.yml` |
| **Algorithm** | Adjusts training difficulty over time — starts with shorter/simpler samples, increases progressively |
| **State** | `archive/curriculum_state.json` (difficulty: 0.3 → 1.0, step: 0.1) |
| **Outputs** | Updated curriculum state; may propose new data ordering for Agent 2 |

### Literature Ingestion Agent — `agents/literature_ingest.py`

| | |
|---|---|
| **Trigger** | Weekly on Sundays at 06:00 UTC OR manual |
| **Workflow** | `.github/workflows/literature-weekly.yml` |
| **API** | Semantic Scholar Academic Graph API (free, no key required) |
| **Keywords** | "state space model", "liquid neural network", "forward-forward learning", "mamba", "selective state space" |
| **Outputs** | Top-N papers (title, abstract snippet, year, citations) stored in `evolutionary_memory.json["literature"]` |

Agent 1 reads these summaries to incorporate cutting-edge ideas into its proposals.

---

## ⚙️ Configuration

All hyperparameters live in `config.yaml`. Agent 1 mutates this file to propose new variants.

### Model

```yaml
model:
  d_model: 192                    # Embedding/hidden dim (divisible by 8)
  n_layers: 4                     # Stacked CDLE blocks
  d_state: 16                     # SSM state dimension
  fractal_levels: 2               # Fractal SSM hierarchy depth
  d_ff: 256                       # Feed-forward expansion
  vocab_size: 256                 # Raw bytes (0–255)
  seq_len: 256                    # Context window (bytes)
  ltc_tau_base: 1.0               # Liquid Time-Constant base
  complexity_gate_threshold: 0.5  # Sparse routing gate
  ff_threshold: 2.0               # Forward-Forward goodness threshold
  ff_variant: "distance"          # "distance" or "contrastive"
  dropout: 0.0                    # Dropout (0 for speed)
  max_params: 12000000            # Hard parameter cap
```

### Training & Data

```yaml
training:
  batch_size: 32
  max_steps: 500          # ~4 min on CPU
  learning_rate: 3.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  warmup_steps: 50

data:
  dataset_name: "roneneldan/TinyStories"
  num_train_samples: 9000
  num_val_samples: 1000
  multi_task:
    piqa_samples: 1000
    gsm8k_samples: 1000
    arc_samples: 1000
```

### Evolution, Curriculum & Energy

```yaml
evolution:
  improvement_threshold: 0.01
  max_history: 20
  qd_archive_path: "archive/qd_population.json"
  qd_max_species: 12
  qd_complexity_bins: 4
  qd_sparsity_bins: 3

curriculum:
  initial_difficulty: 0.3
  max_difficulty: 1.0
  difficulty_step: 0.1
  curriculum_path: "archive/curriculum_state.json"

energy:
  cpu_tdp_watts: 65.0
  flops_weight: 0.5
  watt_weight: 0.5
```

---

## 💸 Cost: $0.00 Forever

| Resource | Provider | Cost |
|----------|----------|------|
| GitHub Actions (`ubuntu-latest`, 2-core CPU) | GitHub | **Free** — unlimited minutes for public repos |
| GitHub Models API (gpt-4o-mini) | GitHub / Azure | **Free** with `GITHUB_TOKEN` |
| Semantic Scholar API | Semantic Scholar | **Free** — no key required for basic use |
| TinyStories dataset | Hugging Face | **Free** |
| PIQA / GSM8K / ARC eval sets | Hugging Face | **Free** |
| GitHub Pages dashboard | GitHub | **Free** for public repos |
| Repository storage | GitHub | **Free** (public repo) |
| **Total** | | **$0.00 / month** |

---

## 🔬 Technical Deep Dive

### CDLE v2 Architecture

```
Input (raw bytes: text, code, binary — vocab_size=256)
      │
      ▼
┌─────────────────────────┐
│  Byte Embedding Layer   │  d_vocab=256, d_model=192
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│          CDLE Block  × n_layers (4)             │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  Fractal Selective SSM                  │    │
│  │  (Mamba-style O(L) + hierarchical state │    │
│  │   across fractal_levels=2)              │    │
│  └────────────────┬────────────────────────┘    │
│                   │                              │
│                   ▼                              │
│  ┌─────────────────────────────────────────┐    │
│  │  Complexity Gate                        │    │
│  │  if complexity < threshold → fast bypass │    │
│  │  else → full Liquid Time-Constant path  │    │
│  └────────────────┬────────────────────────┘    │
│                   │                              │
│                   ▼                              │
│  ┌─────────────────────────────────────────┐    │
│  │  Liquid Time-Constant (LTC) Dynamics    │    │
│  │  Input-adaptive τ gating               │    │
│  └────────────────┬────────────────────────┘    │
│                   │                              │
│                   ▼                              │
│  ┌─────────────────────────────────────────┐    │
│  │  Forward-Forward Update                 │    │
│  │  Variant: Distance-FF or Self-Contrastive│   │
│  │  Layer-local Hebbian learning           │    │
│  └────────────────┬────────────────────────┘    │
│                   │                              │
└───────────────────┼──────────────────────────────┘
                    │
                    ▼
┌─────────────────────────┐
│  Output Head            │  → next-byte logits (256 classes)
└─────────────────────────┘
```

### Forward-Forward Learning Variants

**Distance-FF** (`ff_variant: "distance"`):

```
goodness(h) = Σ(h²)
L_pos = log(1 + exp(-(goodness(h⁺) - θ)))
L_neg = log(1 + exp( (goodness(h⁻) - θ)))
```

**Self-Contrastive** (`ff_variant: "contrastive"`):

```
L = -log(exp(sim(h⁺, h⁺')) / Σ exp(sim(h⁺, hⱼ)))
```

Each layer learns independently — no end-to-end backprop chain required.

### Fractal SSM

The Selective SSM operates at multiple temporal scales simultaneously:

```
Level 0: fine-grained (per-byte dynamics)
Level 1: coarser (aggregated state, longer memory)
  …
Level N: slowest scale (global context)
```

Controlled by `fractal_levels` in `config.yaml`. States are merged hierarchically so the model captures both local patterns and long-range dependencies with O(L) cost.

---

## 🔄 24/7 Operation

| Schedule | Agent | What Happens |
|----------|-------|-------------|
| **Every hour** | Agent 1 | Proposes a new architecture variant via LLM |
| **On push to `test-branch`** | Agent 2 | Trains + multi-task benchmarks the candidate |
| **After benchmark** | Agents 3 + 4 | Route into QD archive + verify stability |
| **After merge** | Agent 5 | Update curriculum difficulty |
| **Every Sunday 06:00 UTC** | Literature Agent | Ingest latest arXiv papers |
| **On push to `main`** | Dashboard | Rebuild & deploy GitHub Pages |

Each generation builds on the best previous architecture. All state is persisted in version-controlled JSON files.

---

## 📜 License

MIT License — completely free to use, modify, and deploy.
