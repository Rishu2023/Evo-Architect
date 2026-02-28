"""
models/baseline_transformer.py
================================
Tiny GPT-2-style causal language model used as a **baseline** for comparison
against the CDLE architecture.

Design:
  * Multi-head causal self-attention (standard scaled dot-product).
  * Pre-LayerNorm (more stable training than post-LN).
  * Learned positional embeddings.
  * Weight-tied input embedding and LM head.
  * Same byte-level (vocab_size=256) interface as CDLEModel.

Parameter count matches the CDLE at default config (~2 M params) so the
comparison is compute-fair.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Standard multi-head causal self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch ≥ 2.0) for efficient fused implementation.

    Args:
        d_model:   Total model dimension.
        n_heads:   Number of attention heads.
        dropout:   Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Project to Q, K, V and split into heads
        qkv = self.qkv_proj(x)                             # (B, L, 3D)
        q, k, v = qkv.split(D, dim=-1)                     # each: (B, L, D)

        # Reshape to (B, n_heads, L, d_head) for multi-head attention
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # PyTorch ≥ 2.0: F.scaled_dot_product_attention supports is_causal flag
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, n_heads, L, d_head)

        # Re-assemble heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_out)


class MLP(nn.Module):
    """
    Standard transformer MLP block: Linear → GELU → Linear.

    Args:
        d_model:  Input/output dimension.
        d_ff:     Intermediate (expansion) dimension.
        dropout:  Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    A single transformer block: Pre-LN attention + Pre-LN MLP with residuals.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        d_ff:     MLP intermediate dimension.
        dropout:  Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class BaselineTransformer(nn.Module):
    """
    Tiny GPT-2-style causal language model for baseline comparison.

    Uses the same byte-level (vocab_size=256) interface as CDLEModel so the
    two models can be swapped with zero code changes in the training loop.

    Args:
        vocab_size:  Token vocabulary size (256 for raw bytes).
        d_model:     Embedding/hidden dimension.
        n_layers:    Number of transformer blocks.
        n_heads:     Number of attention heads per block.
        d_ff:        MLP intermediate dimension.
        seq_len:     Maximum sequence length.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Standard GPT-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        compute_ff_loss: bool = False,   # ignored — kept for API compatibility
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:             (batch, seq_len) integer token indices.
            compute_ff_loss: Ignored (no FF layer in baseline). Kept for API
                             compatibility with CDLEModel.
        Returns:
            (logits, None)
              logits: (batch, seq_len, vocab_size)
        """
        B, L = idx.shape
        assert L <= self.seq_len, f"Sequence length {L} exceeds model max {self.seq_len}"

        positions = torch.arange(L, device=idx.device).unsqueeze(0)
        x = self.dropout(self.embedding(idx) + self.pos_embedding(positions))

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits, None

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict) -> "BaselineTransformer":
        """
        Construct a BaselineTransformer from a config dict.

        Args:
            cfg: Nested config dict (as loaded from config.yaml).
        Returns:
            Initialised BaselineTransformer.
        """
        m = cfg.get("model", cfg)
        # n_heads: largest power of 2 that divides d_model, capped at 8
        d_model = m.get("d_model", 128)
        n_heads = max(h for h in [1, 2, 4, 8] if d_model % h == 0)
        return cls(
            vocab_size=m.get("vocab_size", 256),
            d_model=d_model,
            n_layers=m.get("n_layers", 4),
            n_heads=n_heads,
            d_ff=m.get("d_ff", 256),
            seq_len=m.get("seq_len", 256),
            dropout=m.get("dropout", 0.0),
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = BaselineTransformer.from_config(cfg)
    print(f"Baseline parameter count: {model.count_parameters():,}")

    batch = torch.randint(0, 256, (2, 64))
    logits, _ = model(batch)
    print(f"Output shape: {logits.shape}")
    print("Baseline self-test passed ✓")
