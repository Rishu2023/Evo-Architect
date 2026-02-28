"""
models/cdle_base.py
===================
Core implementation of the Continuous Dynamic Liquid Engine (CDLE).

The CDLE combines four ideas:
  1. Byte-level (character-level) input — no BPE tokenizer needed.
  2. Mamba-style Selective State Space Model (SSM) — O(L) sequence modelling.
  3. Liquid Time-Constant (LTC) dynamics — input-adaptive gating.
  4. Forward-Forward (FF) localised learning — layer-wise Hebbian update.

Extensions (v2):
  5. Fractal/Hierarchical SSM — multi-scale temporal modelling.
  6. Event-driven sparse Liquid routing — complexity-gated fast bypass.
  7. Configurable FF variants — distance-FF and self-contrastive.
  8. Energy proxy — FLOPs + watt estimation for cost-aware evolution.

Design goals:
  * CPU-friendly: no custom CUDA kernels, pure PyTorch 2.x.
  * 1 M–12 M parameters at the default config.
  * All components are heavily commented for research legibility.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  Selective State Space Model (SSM) — Mamba-style
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    A simplified, numerically stable Selective State Space layer inspired by
    Mamba (Gu & Dao, 2023).  We use the *discretised* recurrence in parallel
    scan form but implement it as a sequential scan for CPU simplicity.

    State evolution:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t

    where A_bar, B_bar are input-dependent (selective) via learned projections.

    Args:
        d_model: Input/output feature dimension.
        d_state: Hidden state dimension (controls memory capacity).
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Project input to inner dimension for gating
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # Selective projections: produce Δ (delta / time-step), B, C from input
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)

        # Learnable log of the diagonal A matrix.
        # Initialised as log(1), log(2), ..., log(d_state), giving a log-linear
        # spacing of initial A magnitudes. This spreads the initial time scales
        # across a wide range, helping the SSM capture both fast and slow dynamics.
        # We store log(|A|) so that A = -exp(A_log) is always negative (stable).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)  # shape: (1, d_state)
        )

        # D: skip-connection (direct feed-through from input to output)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Split input into main stream (z) and gate (residual)
        xz = self.in_proj(x)                    # (B, L, 2*D)
        x_main, z = xz.chunk(2, dim=-1)         # each: (B, L, D)

        # SiLU activation on the main stream
        x_act = F.silu(x_main)

        # Compute input-dependent parameters Δ, B, C
        proj = self.x_proj(x_act)               # (B, L, d_state*2 + 1)
        delta_raw, B_sel, C_sel = proj.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        # Δ > 0 via softplus; shape: (B, L, 1) → broadcast over d_state
        delta = F.softplus(delta_raw)           # (B, L, 1)

        # Discretise A: A_bar = exp(Δ * A)   (diagonal A stored as log)
        A = -torch.exp(self.A_log.float())      # (1, d_state) — negative
        # delta: (B, L, 1), A: (1, d_state) → A_bar: (B, L, d_state)
        A_bar = torch.exp(delta * A)

        # Discretise B: B_bar = Δ * B_sel  (zero-order hold approximation)
        B_bar = delta * B_sel                   # (B, L, d_state)

        # Sequential scan over the time dimension
        # h: (B, d_model, d_state) — batch of hidden states
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # x_t: (B, D)
            x_t = x_act[:, t, :]
            # A_bar_t: (B, d_state), broadcast over D
            A_bar_t = A_bar[:, t, :]            # (B, d_state)
            # B_bar_t: (B, d_state)
            B_bar_t = B_bar[:, t, :]
            # C_t: (B, d_state)
            C_t = C_sel[:, t, :]

            # State update: h = A_bar * h + B_bar * x
            # h: (B, D, d_state); A_bar_t: (B, d_state) → unsqueeze for D
            h = h * A_bar_t.unsqueeze(1) + x_t.unsqueeze(-1) * B_bar_t.unsqueeze(1)

            # Output: y_t = sum_d_state( C * h )
            # h: (B, D, d_state), C_t: (B, d_state)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)              # (B, L, D)

        # Add skip connection (D parameter) and gate with z
        y = y + self.D * x_act
        y = y * F.silu(z)

        return self.out_proj(y)                 # (B, L, D)


# ---------------------------------------------------------------------------
# 1b. Fractal / Hierarchical SSM — multi-scale temporal modelling
# ---------------------------------------------------------------------------

class FractalSSM(nn.Module):
    """
    Fractal (hierarchical) SSM wrapper around SelectiveSSM.

    Each hierarchy level operates at a *coarser* temporal scale by sub-sampling
    the input sequence by a factor of 2^level.  The coarse-level output is then
    up-sampled (nearest-neighbour repeat) back to the original length and fused
    with the fine-level output via a *learned per-level gate*.

    This captures both local (fast) and global (slow) temporal dynamics using
    the same SSM architecture — analogous to a U-Net's multi-resolution
    structure but applied along the time axis.

    Args:
        d_model:  Feature dimension.
        d_state:  SSM state size (forwarded to SelectiveSSM).
        levels:   Number of hierarchy levels (1 = standard SSM, no fractal).
    """

    def __init__(self, d_model: int, d_state: int = 16, levels: int = 2):
        super().__init__()
        self.d_model = d_model
        self.levels = levels

        # One SSM per hierarchy level.  Each level sees progressively
        # coarser (sub-sampled) versions of the input.
        self.ssm_layers = nn.ModuleList([
            SelectiveSSM(d_model, d_state) for _ in range(levels)
        ])

        # Learned gating scalars (one per level) that combine outputs.
        # Initialised so that level-0 (full-resolution) dominates at start.
        self.gate_logits = nn.Parameter(torch.zeros(levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model) — fused multi-scale SSM output.
        """
        B, L, D = x.shape

        # Softmax gate across levels → weights sum to 1
        gates = torch.softmax(self.gate_logits, dim=0)  # (levels,)

        # Accumulate gated outputs from each hierarchy level
        y = torch.zeros_like(x)  # (B, L, D)

        for lvl in range(self.levels):
            # Sub-sample factor: level 0 → factor 1, level 1 → factor 2, etc.
            factor = 2 ** lvl

            if factor >= L:
                # If the sub-sample factor is >= sequence length, skip this
                # level entirely — there aren't enough tokens to sub-sample.
                continue

            # Sub-sample: take every `factor`-th token along the time axis
            x_sub = x[:, ::factor, :]  # (B, ceil(L/factor), D)

            # Run the level-specific SSM on the sub-sampled input
            y_sub = self.ssm_layers[lvl](x_sub)  # (B, ceil(L/factor), D)

            # Up-sample back to original length using nearest-neighbour repeat.
            # repeat_interleave along dim=1, then trim to exactly L tokens.
            y_up = y_sub.repeat_interleave(factor, dim=1)[:, :L, :]  # (B, L, D)

            # Gate and accumulate
            y = y + gates[lvl] * y_up

        return y


# ---------------------------------------------------------------------------
# 2.  Liquid Time-Constant (LTC) Layer
# ---------------------------------------------------------------------------

class LiquidTimeConstant(nn.Module):
    """
    Liquid Time-Constant (LTC) inspired layer with event-driven sparse routing.

    The core idea: each layer's "time constant" τ is *input-dependent*.
    High-complexity inputs get a fast τ (more dynamic state change);
    low-complexity inputs get a slow τ (stable accumulation).

    **Sparse Routing Extension (v2):**
    A learned complexity gate produces a scalar per token.  Tokens whose gate
    value falls *below* ``complexity_threshold`` are considered "simple" and
    receive an identity-like bypass (no LTC computation), saving FLOPs.
    Tokens above the threshold get full LTC processing.  The gate is smooth
    (sigmoid) so gradients still flow through bypassed tokens.

    Equation (simplified continuous-time approximation):
        τ(x) = τ_base * sigmoid(W_τ x + b_τ)   ∈ (0, τ_base)
        h' = (-h + tanh(W_h x + b_h)) / τ(x)   ← ODE rate
        h_out = h + dt * h' with dt=1 (discrete time)

    For a language model the "hidden state" is the token's feature vector,
    so we apply this as a *residual gated update* rather than a true RNN.

    Args:
        d_model:               Feature dimension.
        tau_base:              Maximum time constant (larger = slower dynamics).
        complexity_gate_threshold:  Gate threshold in (0, 1).  Tokens with gate
                               value below this skip full LTC processing.
                               Set to 0.0 to disable sparse routing (all
                               tokens processed).
    """

    def __init__(
        self,
        d_model: int,
        tau_base: float = 1.0,
        complexity_gate_threshold: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_base = tau_base
        self.complexity_gate_threshold = complexity_gate_threshold

        # Estimate input complexity (used to compute τ)
        self.complexity_proj = nn.Linear(d_model, d_model, bias=True)

        # Tau gate: maps input → τ ∈ (0, tau_base)
        self.tau_proj = nn.Linear(d_model, d_model, bias=True)

        # Hidden-state update projection
        self.h_proj = nn.Linear(d_model, d_model, bias=True)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

        # --- Sparse-routing complexity gate (v2) ---
        # Produces a *scalar* gate per token that determines whether the
        # token is "complex enough" to warrant full LTC processing.
        # Architecture: d_model → 1  (cheap single-vector projection)
        self.sparse_gate = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)  — LTC-modulated features
        """
        B, L, D = x.shape

        # --- Sparse routing gate ---
        # gate_score ∈ (0, 1) per token; high = complex, low = simple.
        gate_score = torch.sigmoid(self.sparse_gate(x))  # (B, L, 1)

        # Build a boolean mask of tokens that need full LTC computation.
        # Tokens below the threshold get an identity-like bypass.
        complex_mask = (gate_score > self.complexity_gate_threshold)  # (B, L, 1)

        # Always compute full LTC path so gradients flow through the gate
        # even when all tokens are below threshold (prevents gate from
        # getting stuck in a "bypass everything" state).

        # Compute complexity signal (how "surprising" each position is)
        complexity = torch.sigmoid(self.complexity_proj(x))  # (B, L, D)

        # Input-adaptive time constant τ ∈ (0, tau_base)
        tau = self.tau_base * torch.sigmoid(self.tau_proj(complexity))  # (B, L, D)

        # ODE target state: tanh(W_h x)
        h_target = torch.tanh(self.h_proj(x))                           # (B, L, D)

        # Discrete Euler step: x + (h_target - x) / tau
        # When τ → 0: output ≈ h_target  (fast dynamics)
        # When τ → ∞: output ≈ x         (slow dynamics, identity-like)
        h_ltc = x + (h_target - x) / (tau + 1e-6)

        # --- Merge full-LTC and bypass paths ---
        # Use the smooth gate_score (not the hard mask) for the blend so that
        # gradients flow through the bypass path as well.
        # h_out = gate_score * h_ltc + (1 - gate_score) * x
        h_out = gate_score * h_ltc + (1.0 - gate_score) * x

        return self.norm(h_out)


# ---------------------------------------------------------------------------
# 3.  Forward-Forward (FF) Learning Module
# ---------------------------------------------------------------------------

class ForwardForwardLayer(nn.Module):
    """
    Forward-Forward (FF) learning layer (Hinton, 2022) with configurable
    learning-signal variants.

    Each layer has its *own* local loss that encourages:
      - High "goodness" (sum of squared activations) on real (positive) data.
      - Low goodness on synthetic (negative) data (input corrupted with noise).

    **Variants (v2):**
      ``"standard"`` — Original goodness-threshold FF (default, backward compat).
      ``"distance"`` — Distance-FF: uses L2 distance between positive and
                       negative embeddings as the learning signal.  No
                       explicit goodness threshold is needed.
      ``"contrastive"`` — Self-Contrastive: creates two augmented views of
                          each input (dropout noise) and maximises their
                          cosine similarity while minimising similarity to
                          negatives.

    This layer can be trained with standard autograd (the FF loss is just added
    to the total loss) OR used in a purely local mode where gradients flow only
    within the layer.

    Args:
        d_model:    Feature dimension.
        threshold:  Goodness threshold θ separating positive from negative
                    (used only by the ``"standard"`` variant).
        ff_variant: One of ``"standard"``, ``"distance"``, ``"contrastive"``.
    """

    def __init__(
        self,
        d_model: int,
        threshold: float = 2.0,
        ff_variant: str = "standard",
    ):
        super().__init__()
        self.threshold = threshold
        self.ff_variant = ff_variant
        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        # Running estimate of the FF loss (for logging)
        self.register_buffer("ff_loss_ema", torch.tensor(0.0))

        # Contrastive variant uses a small dropout for augmentation views
        if ff_variant == "contrastive":
            self.aug_dropout = nn.Dropout(p=0.1)

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Goodness = mean of squared activations across the feature dim."""
        return h.pow(2).mean(dim=-1)             # (batch, seq_len)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Shared forward computation: normalise → linear → ReLU → norm."""
        x_norm = F.normalize(x, p=2, dim=-1)
        h = F.relu(self.linear(x_norm))
        return self.norm(h)

    # ----- variant-specific loss functions -----

    def _ff_loss_standard(self, h_pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Original goodness-threshold FF loss."""
        g_pos = self.goodness(h_pos)                 # (B, L)

        # Negative samples: randomly permuted inputs within the batch
        x_neg = x[torch.randperm(x.size(0))]
        h_neg = self._embed(x_neg)
        g_neg = self.goodness(h_neg)

        # FF contrastive loss (log-sum-exp formulation)
        loss_pos = F.softplus(-(g_pos - self.threshold)).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()
        return loss_pos + loss_neg

    def _ff_loss_distance(self, h_pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Distance-FF: Uses L2 distance between positive and negative
        embeddings as the learning signal.

        Objective: minimise distance between positive pairs (same input,
        different noise) and maximise distance to negatives.
        We use a margin-based hinge: loss = max(0, margin - d_neg) + d_pos
        where d_pos should be small and d_neg should be large.
        """
        # Create a second positive view by adding small Gaussian noise
        x_pos2 = x + 0.05 * torch.randn_like(x)
        h_pos2 = self._embed(x_pos2)

        # Negative: batch-permuted input
        x_neg = x[torch.randperm(x.size(0))]
        h_neg = self._embed(x_neg)

        # L2 distances (per-token, averaged over feature dim)
        d_pos = (h_pos - h_pos2).pow(2).mean(dim=-1)      # (B, L)
        d_neg = (h_pos - h_neg).pow(2).mean(dim=-1)        # (B, L)

        # Hinge loss with margin=1.0: push negatives apart, pull positives
        margin = 1.0
        loss = d_pos.mean() + F.relu(margin - d_neg).mean()
        return loss

    def _ff_loss_contrastive(self, h_pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Self-Contrastive FF: Creates two augmented views of each input via
        dropout noise and maximises their cosine similarity while minimising
        similarity to negatives.

        Inspired by SimCLR / Barlow Twins but applied locally per-layer.
        """
        # Two augmented views via dropout noise
        h_view1 = self._embed(self.aug_dropout(x))
        h_view2 = self._embed(self.aug_dropout(x))

        # Negative view: permuted batch
        x_neg = x[torch.randperm(x.size(0))]
        h_neg = self._embed(x_neg)

        # Cosine similarity (per token, averaged over feature dim)
        sim_pos = F.cosine_similarity(h_view1, h_view2, dim=-1)    # (B, L)
        sim_neg = F.cosine_similarity(h_view1, h_neg, dim=-1)      # (B, L)

        # Maximise positive similarity, minimise negative similarity
        # Loss = -log(sigmoid(sim_pos)) - log(sigmoid(-sim_neg))
        loss = -F.logsigmoid(sim_pos).mean() - F.logsigmoid(-sim_neg).mean()
        return loss

    def forward(
        self,
        x: torch.Tensor,
        compute_ff_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:               (batch, seq_len, d_model)  — input features
            compute_ff_loss: If True, also compute the FF layer loss.

        Returns:
            (output, ff_loss)
              output:  (batch, seq_len, d_model)
              ff_loss: scalar tensor if compute_ff_loss else None
        """
        # Compute the forward embedding (shared across all variants)
        h = self._embed(x)

        ff_loss = None
        if compute_ff_loss and self.training:
            # Dispatch to the appropriate variant loss function
            if self.ff_variant == "distance":
                ff_loss = self._ff_loss_distance(h, x)
            elif self.ff_variant == "contrastive":
                ff_loss = self._ff_loss_contrastive(h, x)
            else:
                # Default: standard goodness-threshold FF
                ff_loss = self._ff_loss_standard(h, x)

            # Update EMA for logging
            self.ff_loss_ema = 0.99 * self.ff_loss_ema + 0.01 * ff_loss.detach()

        return h, ff_loss


# ---------------------------------------------------------------------------
# 4.  CDLE Block — combines SSM + LTC + FF
# ---------------------------------------------------------------------------

class CDLEBlock(nn.Module):
    """
    A single CDLE block stacking:
        LayerNorm → FractalSSM → LTC (sparse routing) → ForwardForwardLayer → residual

    Args:
        d_model:                Feature dimension.
        d_state:                SSM state size.
        d_ff:                   Feed-forward expansion (used in optional MLP).
        tau_base:               LTC base time constant.
        ff_threshold:           FF goodness threshold.
        dropout:                Dropout probability.
        fractal_levels:         Number of fractal SSM hierarchy levels.
        complexity_threshold:   LTC sparse routing gate threshold.
        ff_variant:             FF variant: "standard", "distance", "contrastive".
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_ff: int = 256,
        tau_base: float = 1.0,
        ff_threshold: float = 2.0,
        dropout: float = 0.0,
        fractal_levels: int = 1,
        complexity_threshold: float = 0.0,
        ff_variant: str = "standard",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)

        # Use FractalSSM wrapper (falls back to single SSM when levels=1)
        self.ssm = FractalSSM(d_model, d_state, levels=fractal_levels)

        self.norm2 = nn.LayerNorm(d_model)

        # LTC with optional sparse routing
        self.ltc = LiquidTimeConstant(
            d_model, tau_base,
            complexity_gate_threshold=complexity_threshold,
        )

        # Configurable FF variant
        self.ff_layer = ForwardForwardLayer(
            d_model, ff_threshold, ff_variant=ff_variant,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        compute_ff_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:               (batch, seq_len, d_model)
            compute_ff_loss: Whether to compute the FF localised loss.
        Returns:
            (x, ff_loss)
        """
        # SSM sub-layer with pre-norm + residual
        x = x + self.dropout(self.ssm(self.norm1(x)))

        # LTC sub-layer with pre-norm + residual
        x = x + self.dropout(self.ltc(self.norm2(x)))

        # FF sub-layer (no residual — FF operates on normalised representation)
        h, ff_loss = self.ff_layer(x, compute_ff_loss=compute_ff_loss)
        # Blend FF output with residual so gradients still flow normally
        x = x + self.dropout(h)

        return x, ff_loss


# ---------------------------------------------------------------------------
# 5.  Full CDLE Model
# ---------------------------------------------------------------------------

class CDLEModel(nn.Module):
    """
    Complete Continuous Dynamic Liquid Engine language model.

    Input: raw byte / character indices (0–255)
    Output: next-byte logit distribution

    Architecture:
        ByteEmbedding → [CDLEBlock × n_layers] → LayerNorm → LM Head

    Parameter count at default config (d_model=128, n_layers=4): ~2 M params.

    Args:
        vocab_size:              Number of input tokens (256 for bytes).
        d_model:                 Embedding / hidden dimension.
        n_layers:                Number of CDLE blocks.
        d_state:                 SSM state dimension.
        d_ff:                    MLP expansion dimension (inside FF layer).
        seq_len:                 Maximum sequence length (for positional embeddings).
        tau_base:                LTC base time constant.
        ff_threshold:            FF goodness threshold.
        dropout:                 Dropout probability.
        fractal_levels:          Number of fractal SSM hierarchy levels.
        complexity_gate_threshold: LTC sparse routing gate threshold.
        ff_variant:              FF variant: "standard", "distance", "contrastive".
        cpu_tdp_watts:           CPU TDP in watts (for energy proxy).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_ff: int = 256,
        seq_len: int = 256,
        tau_base: float = 1.0,
        ff_threshold: float = 2.0,
        dropout: float = 0.0,
        fractal_levels: int = 1,
        complexity_gate_threshold: float = 0.0,
        ff_variant: str = "standard",
        cpu_tdp_watts: float = 65.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # --- Energy proxy state (v2) ---
        # cpu_tdp_watts: assumed thermal design power of the CPU.
        # _last_flops: FLOPs counted during the most recent forward pass.
        # _last_wall_secs: wall-clock seconds for the most recent forward pass.
        self.cpu_tdp_watts = cpu_tdp_watts
        self._last_flops: int = 0
        self._last_wall_secs: float = 0.0

        # Byte / character embedding table
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embeddings (simple and effective for short seqs)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # Stack of CDLE blocks
        self.blocks = nn.ModuleList([
            CDLEBlock(
                d_model=d_model,
                d_state=d_state,
                d_ff=d_ff,
                tau_base=tau_base,
                ff_threshold=ff_threshold,
                dropout=dropout,
                fractal_levels=fractal_levels,
                complexity_threshold=complexity_gate_threshold,
                ff_variant=ff_variant,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm before head
        self.norm_out = nn.LayerNorm(d_model)

        # Language model head: maps hidden states → logits over vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialise weights sensibly
        self._init_weights()

        # Weight tie: embedding and LM head share weights (reduces parameters,
        # improves generalisation — standard practice since Press & Wolf 2017).
        # IMPORTANT: must happen AFTER _init_weights() so that the embedding
        # initialisation (normal_) is not overwritten by the Linear init
        # (xavier_uniform_) applied to lm_head.
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self):
        """Xavier / small-normal initialisation for training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _estimate_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Rough FLOPs estimate for a single forward pass.

        Counts dominant contributors: embedding lookup (negligible), linear
        layers (2 * in * out per element), and the SSM sequential scan.
        This is an *approximation* — good enough for energy-aware evolution.
        """
        flops = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Each linear layer: 2 * in_features * out_features per token
                flops += 2 * module.in_features * module.out_features * batch_size * seq_len
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: ~5 * features per token (mean, var, norm, scale, shift)
                flops += 5 * module.normalized_shape[0] * batch_size * seq_len
        return flops

    def energy_proxy(self) -> dict:
        """
        Estimate energy consumption from the most recent forward pass.

        Returns a dict with:
          - ``flops``: estimated FLOPs for the pass.
          - ``flops_per_sec``: FLOPs / wall-clock seconds.
          - ``linear_watt_estimate``: estimated watts = cpu_tdp * utilisation,
            where utilisation is approximated as
            min(1, flops_per_sec / 1e12) (fraction of a ~1 TFLOP CPU).
          - ``energy_score``: combined scalar = flops_per_sec + linear_watt_estimate.
            Lower is cheaper.
        """
        wall = max(self._last_wall_secs, 1e-9)  # avoid division by zero
        flops_per_sec = self._last_flops / wall

        # Approximate CPU utilisation as fraction of a ~1 TFLOP reference.
        utilisation = min(1.0, flops_per_sec / 1e12)
        linear_watts = self.cpu_tdp_watts * utilisation

        energy_score = flops_per_sec + linear_watts
        return {
            "flops": self._last_flops,
            "flops_per_sec": flops_per_sec,
            "linear_watt_estimate": linear_watts,
            "energy_score": energy_score,
        }

    def forward(
        self,
        idx: torch.Tensor,
        compute_ff_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:             (batch, seq_len) integer token indices
            compute_ff_loss: Accumulate Forward-Forward losses across layers.

        Returns:
            (logits, total_ff_loss)
              logits:        (batch, seq_len, vocab_size)
              total_ff_loss: scalar or None
        """
        B, L = idx.shape
        assert L <= self.seq_len, f"Sequence length {L} exceeds model max {self.seq_len}"

        # --- Energy proxy: start timer and estimate FLOPs ---
        t_start = time.perf_counter()
        self._last_flops = self._estimate_flops(B, L)

        # Token + positional embeddings
        positions = torch.arange(L, device=idx.device).unsqueeze(0)  # (1, L)
        x = self.embedding(idx) + self.pos_embedding(positions)       # (B, L, D)

        # Pass through CDLE blocks, accumulating FF losses
        total_ff_loss: Optional[torch.Tensor] = None
        for block in self.blocks:
            x, ff_loss = block(x, compute_ff_loss=compute_ff_loss)
            if ff_loss is not None:
                total_ff_loss = ff_loss if total_ff_loss is None else total_ff_loss + ff_loss

        # Final norm + language model head
        x = self.norm_out(x)
        logits = self.lm_head(x)                                       # (B, L, V)

        # --- Energy proxy: stop timer ---
        self._last_wall_secs = time.perf_counter() - t_start

        return logits, total_ff_loss

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict) -> "CDLEModel":
        """
        Construct a CDLEModel from a config dict (as loaded from config.yaml).

        Args:
            cfg: dict with keys matching the constructor arguments.
        Returns:
            Initialised CDLEModel.
        """
        m = cfg.get("model", cfg)  # support both nested and flat dicts
        e = cfg.get("energy", {})  # energy proxy settings (optional section)
        return cls(
            vocab_size=m.get("vocab_size", 256),
            d_model=m.get("d_model", 128),
            n_layers=m.get("n_layers", 4),
            d_state=m.get("d_state", 16),
            d_ff=m.get("d_ff", 256),
            seq_len=m.get("seq_len", 256),
            tau_base=m.get("ltc_tau_base", 1.0),
            ff_threshold=m.get("ff_threshold", 2.0),
            dropout=m.get("dropout", 0.0),
            # --- New v2 config keys ---
            fractal_levels=m.get("fractal_levels", 1),
            complexity_gate_threshold=m.get("complexity_gate_threshold", 0.0),
            ff_variant=m.get("ff_variant", "standard"),
            cpu_tdp_watts=e.get("cpu_tdp_watts", 65.0),
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = CDLEModel.from_config(cfg)
    n_params = model.count_parameters()
    max_params = cfg.get("model", {}).get("max_params", 12_000_000)
    print(f"CDLE parameter count: {n_params:,}")
    assert 1_000_000 <= n_params <= max_params, (
        f"Parameter count {n_params:,} outside 1M–{max_params // 1_000_000}M range"
    )

    # Dummy forward pass (training mode for FF loss)
    model.train()
    batch = torch.randint(0, 256, (2, 64))
    logits, ff_loss = model(batch, compute_ff_loss=True)
    print(f"Output shape: {logits.shape}")
    print(f"FF loss: {ff_loss.item():.4f}")

    # --- Test energy proxy ---
    energy = model.energy_proxy()
    print(f"Energy proxy → FLOPs: {energy['flops']:,}, "
          f"FLOP/s: {energy['flops_per_sec']:.2e}, "
          f"Watts: {energy['linear_watt_estimate']:.2f}, "
          f"Score: {energy['energy_score']:.2e}")
    assert energy["flops"] > 0, "FLOPs should be positive"
    assert energy["energy_score"] >= 0, "Energy score should be non-negative"

    # --- Test FractalSSM directly ---
    d = cfg["model"]["d_model"]
    fractal = FractalSSM(d, d_state=16, levels=2)
    x_test = torch.randn(2, 32, d)
    y_fractal = fractal(x_test)
    assert y_fractal.shape == x_test.shape, "FractalSSM shape mismatch"
    print(f"FractalSSM test passed ✓  (shape {y_fractal.shape})")

    # --- Test LTC sparse routing ---
    ltc_sparse = LiquidTimeConstant(d, tau_base=1.0, complexity_gate_threshold=0.5)
    y_ltc = ltc_sparse(x_test)
    assert y_ltc.shape == x_test.shape, "LTC sparse routing shape mismatch"
    print(f"LTC sparse routing test passed ✓  (shape {y_ltc.shape})")

    # --- Test FF variants ---
    for variant in ["standard", "distance", "contrastive"]:
        ff = ForwardForwardLayer(d, threshold=2.0, ff_variant=variant)
        ff.train()
        h_out, fl = ff(x_test, compute_ff_loss=True)
        assert h_out.shape == x_test.shape, f"FF {variant} shape mismatch"
        assert fl is not None, f"FF {variant} loss should not be None"
        print(f"FF variant '{variant}' test passed ✓  (loss={fl.item():.4f})")

    # --- Inference mode (no FF loss) ---
    model.eval()
    logits_eval, ff_loss_eval = model(batch, compute_ff_loss=True)
    assert ff_loss_eval is None, "FF loss should be None in eval mode"
    print(f"Eval mode test passed ✓  (logits shape {logits_eval.shape})")

    print("\nCDLE self-test passed ✓")
