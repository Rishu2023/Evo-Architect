"""
models/cdle_base.py
===================
Core implementation of the Continuous Dynamic Liquid Engine (CDLE).

The CDLE combines four ideas:
  1. Byte-level (character-level) input — no BPE tokenizer needed.
  2. Mamba-style Selective State Space Model (SSM) — O(L) sequence modelling.
  3. Liquid Time-Constant (LTC) dynamics — input-adaptive gating.
  4. Forward-Forward (FF) localised learning — layer-wise Hebbian update.

Design goals:
  * CPU-friendly: no custom CUDA kernels, pure PyTorch 2.x.
  * 1 M–8 M parameters at the default config.
  * All components are heavily commented for research legibility.
"""

import math
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
# 2.  Liquid Time-Constant (LTC) Layer
# ---------------------------------------------------------------------------

class LiquidTimeConstant(nn.Module):
    """
    Liquid Time-Constant (LTC) inspired layer.

    The core idea: each layer's "time constant" τ is *input-dependent*.
    High-complexity inputs get a fast τ (more dynamic state change);
    low-complexity inputs get a slow τ (stable accumulation).

    Equation (simplified continuous-time approximation):
        τ(x) = τ_base * sigmoid(W_τ x + b_τ)   ∈ (0, τ_base)
        h' = (-h + tanh(W_h x + b_h)) / τ(x)   ← ODE rate
        h_out = h + dt * h' with dt=1 (discrete time)

    For a language model the "hidden state" is the token's feature vector,
    so we apply this as a *residual gated update* rather than a true RNN.

    Args:
        d_model:   Feature dimension.
        tau_base:  Maximum time constant (larger = slower dynamics).
    """

    def __init__(self, d_model: int, tau_base: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.tau_base = tau_base

        # Estimate input complexity (used to compute τ)
        self.complexity_proj = nn.Linear(d_model, d_model, bias=True)

        # Tau gate: maps input → τ ∈ (0, tau_base)
        self.tau_proj = nn.Linear(d_model, d_model, bias=True)

        # Hidden-state update projection
        self.h_proj = nn.Linear(d_model, d_model, bias=True)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)  — LTC-modulated features
        """
        # Compute complexity signal (how "surprising" each position is)
        complexity = torch.sigmoid(self.complexity_proj(x))  # (B, L, D)

        # Input-adaptive time constant τ ∈ (0, tau_base)
        tau = self.tau_base * torch.sigmoid(self.tau_proj(complexity))  # (B, L, D)

        # ODE target state: tanh(W_h x)
        h_target = torch.tanh(self.h_proj(x))                           # (B, L, D)

        # Discrete Euler step: x + (h_target - x) / tau
        # When τ → 0: output ≈ h_target  (fast dynamics)
        # When τ → ∞: output ≈ x         (slow dynamics, identity-like)
        h_out = x + (h_target - x) / (tau + 1e-6)

        return self.norm(h_out)


# ---------------------------------------------------------------------------
# 3.  Forward-Forward (FF) Learning Module
# ---------------------------------------------------------------------------

class ForwardForwardLayer(nn.Module):
    """
    Forward-Forward (FF) learning layer (Hinton, 2022).

    Each layer has its *own* local loss that encourages:
      - High "goodness" (sum of squared activations) on real (positive) data.
      - Low goodness on synthetic (negative) data (input corrupted with noise).

    This layer can be trained with standard autograd (the FF loss is just added
    to the total loss) OR used in a purely local mode where gradients flow only
    within the layer.

    Args:
        d_model:    Feature dimension.
        threshold:  Goodness threshold θ separating positive from negative.
    """

    def __init__(self, d_model: int, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        # Running estimate of the FF loss (for logging)
        self.register_buffer("ff_loss_ema", torch.tensor(0.0))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Goodness = mean of squared activations across the feature dim."""
        return h.pow(2).mean(dim=-1)             # (batch, seq_len)

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
        # Normalise input so goodness is scale-invariant (per Hinton 2022)
        x_norm = F.normalize(x, p=2, dim=-1)
        h = F.relu(self.linear(x_norm))
        h = self.norm(h)

        ff_loss = None
        if compute_ff_loss and self.training:
            # Positive samples: real data
            g_pos = self.goodness(h)                 # (B, L)

            # Negative samples: randomly permuted inputs within the batch
            x_neg = x[torch.randperm(x.size(0))]
            x_neg_norm = F.normalize(x_neg, p=2, dim=-1)
            h_neg = F.relu(self.linear(x_neg_norm))
            h_neg = self.norm(h_neg)
            g_neg = self.goodness(h_neg)

            # FF contrastive loss (log-sum-exp formulation)
            loss_pos = F.softplus(-(g_pos - self.threshold)).mean()
            loss_neg = F.softplus(g_neg - self.threshold).mean()
            ff_loss = loss_pos + loss_neg

            # Update EMA for logging
            self.ff_loss_ema = 0.99 * self.ff_loss_ema + 0.01 * ff_loss.detach()

        return h, ff_loss


# ---------------------------------------------------------------------------
# 4.  CDLE Block — combines SSM + LTC + FF
# ---------------------------------------------------------------------------

class CDLEBlock(nn.Module):
    """
    A single CDLE block stacking:
        LayerNorm → SelectiveSSM → LTC → ForwardForwardLayer → residual

    Args:
        d_model:    Feature dimension.
        d_state:    SSM state size.
        d_ff:       Feed-forward expansion (used in optional MLP).
        tau_base:   LTC base time constant.
        ff_threshold: FF goodness threshold.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_ff: int = 256,
        tau_base: float = 1.0,
        ff_threshold: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ltc = LiquidTimeConstant(d_model, tau_base)
        self.ff_layer = ForwardForwardLayer(d_model, ff_threshold)
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
        vocab_size:    Number of input tokens (256 for bytes).
        d_model:       Embedding / hidden dimension.
        n_layers:      Number of CDLE blocks.
        d_state:       SSM state dimension.
        d_ff:          MLP expansion dimension (inside FF layer).
        seq_len:       Maximum sequence length (for positional embeddings).
        tau_base:      LTC base time constant.
        ff_threshold:  FF goodness threshold.
        dropout:       Dropout probability.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

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
    print(f"CDLE parameter count: {model.count_parameters():,}")

    # Dummy forward pass
    batch = torch.randint(0, 256, (2, 64))
    logits, ff_loss = model(batch, compute_ff_loss=True)
    print(f"Output shape: {logits.shape}")
    print(f"FF loss: {ff_loss.item():.4f}")
    print("CDLE self-test passed ✓")
