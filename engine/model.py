"""Paged language-model interface and a deterministic toy implementation.

``PagedLM`` is the contract the scheduler drives and the Qwen adapter implements:
it exposes two forward primitives over a :class:`PagedKVCache`,

* ``prefill`` -- feed a ragged prompt chunk, write all its K/V into physical
  blocks, return the next-token logits of every row;
* ``decode`` -- feed one freshly sampled token per row, write its K/V, return the
  logits for the token following it.

Both share the same per-layer stack (RMSNorm -> QKV -> RoPE -> *paged attention*
-> O -> residual -> MLP -> residual), which is what makes the from-scratch claim
checkable: the toy model below is the *oracle harness* that proves the engine
logic, while ``engine/qwen.py`` reuses the exact same primitives on real weights.

Rotary positional embeddings are implemented by hand (``make_rotary_cache`` /
``apply_rotary``) so toy and real adapter share one code path and the real
(model-loaded) one can be validated token-for-token against ``transformers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .attention import paged_attention_decode, paged_attention_prefill_chunk
from .kv import PagedKVCache
from .utils import setup_logger

logger = setup_logger(__name__)


# --------------------------------------------------------------------- rotary
def make_rotary_cache(head_dim: int, theta: float, max_len: int, device: torch.device) -> torch.Tensor:
    """Precompute the interleaved RoPE ``cos``/``sin`` table.

    Returns ``(max_len, 2, head_dim)``; ``[:, 0]`` is ``cos``, ``[:, 1]`` is
    ``sin``. The angle frequency duplicates its last dimension to match
    ``transformers``' ``rotate_half`` convention.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(-1)
    freqs = positions * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    table = torch.stack((emb.cos(), emb.sin()), dim=1)
    return table.to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------- norms
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the statistic in fp32 for numerical stability (as ``transformers``
        # does) but cast the output back to the input dtype so fp16 activations stay
        # fp16 and float32 weights remain usable.
        input_dtype = x.dtype
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x.float() / rms) * self.weight.float()).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


# ------------------------------------------------------------- decoder layer
class PagedDecoderLayer(nn.Module):
    """A decoder block with paged (gather-based) self-attention + SwiGLU MLP.

    Q/K/V projections and RoPE application live here; ``decode_step`` and
    ``prefill_step`` route into the two paged-attention entry points and share the
    residual structure that the Qwen adapter reuses verbatim.
    """

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate: int,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.input_layernorm = RMSNorm(hidden)
        self.q = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        self.k = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = SwiGLU(hidden, intermediate)

    def decode_step(
        self,
        hidden: torch.Tensor,
        kv: PagedKVCache,
        layer_index: int,
        seq_ids: Sequence[int],
        ctx_lengths: Sequence[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one new token per row; return ``(B, hidden)``."""
        B = hidden.shape[0]
        residual = hidden
        x = self.input_layernorm(hidden)
        q_raw = self.q(x).view(B, self.num_q_heads, self.head_dim)
        k_raw = self.k(x).view(B, self.num_kv_heads, self.head_dim)
        v_raw = self.v(x).view(B, self.num_kv_heads, self.head_dim)
        q = apply_rotary(q_raw, sin, cos)
        k = apply_rotary(k_raw, sin, cos)
        # Cache the queried token's own K/V at position ctx_lengths[b] FIRST ...
        for b in range(B):
            kv.write(layer_index, seq_ids[b], int(ctx_lengths[b]), k[b], v_raw[b])
        # ... then attend EVERY position up to and including itself (causal
        # position t sees 0..t). Gathering ctx+1 stored cells covers the just
        # written self position too, matching the reference decoder exactly.
        attend_lengths = [int(c) + 1 for c in ctx_lengths]
        out = paged_attention_decode(q, kv, layer_index, seq_ids, attend_lengths, self.num_q_heads)
        hidden = residual + self.o(out.view(B, -1))
        residual = hidden
        hidden = residual + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


# ----------------------------------------------------------------- plain text
class ToyTokenizer:
    """Character-level tokenizer covering ``vocab_size`` symbols."""

    def __init__(self, vocab_size: int = 16) -> None:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.vocab_size = vocab_size
        self.itos = list(alphabet[:vocab_size])
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}
        self.pad_token_id = 0
        self.eos_token_id = None

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.itos[int(i) % self.vocab_size] for i in token_ids)


# ----------------------------------------------------------------- toy model
@dataclass
class ToyConfig:
    vocab_size: int = 16
    hidden_size: int = 64
    num_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    max_position_embeddings: int = 256
    theta: float = 10000.0
    logit_scale: float = 6.0
    seed: int = 0


class ToyPagedLM(nn.Module):
    """Deterministic, memory-lean transformer consuming ``PagedKVCache``."""

    def __init__(self, cfg: ToyConfig, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.tokenizer = ToyTokenizer(cfg.vocab_size)
        self.device = device
        self.num_layers = cfg.num_layers
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self._cos_sin = make_rotary_cache(
            cfg.head_dim, cfg.theta, cfg.max_position_embeddings, device
        )
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                PagedDecoderLayer(
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.intermediate_size,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.logit_scale != 1.0:
            with torch.no_grad():
                self.lm_head.weight.mul_(cfg.logit_scale)
        # Config is exposed so the scheduler can build a PagedKVCache to size.
        self.config = cfg
        self.forward_calls = 0
        # The modules above are constructed on CPU; move the whole stack (embeddings,
        # layers, norms, head) to the target device so input tensors and weights agree.
        self.to(device=device)
        logger.info("toy paged model: %d layers, %d q heads, %d kv heads, d=%d",
                    cfg.num_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    def reset_counters(self) -> None:
        self.forward_calls = 0

    # ----------------------------------------------------------- paged api
    def prefill(
        self,
        prompt_2d: torch.Tensor,
        kv: PagedKVCache,
        seq_ids: Sequence[int],
        seq_lengths: Sequence[int],
    ) -> torch.Tensor:
        """Prefill a battery of prompts; return first-token logits ``(B, V)``."""
        self.forward_calls += 1
        B, chunk = prompt_2d.shape
        hidden = self.embed_tokens(prompt_2d)          # (B, chunk, H)
        positions = torch.arange(chunk, dtype=torch.long, device=self.device)
        cos = self._cos_sin[positions, 0]              # (chunk, d)
        sin = self._cos_sin[positions, 1]
        for index, layer in enumerate(self.layers):
            hidden = self._prefill_layer(
                hidden, layer, index, kv, seq_ids, seq_lengths, cos, sin
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)                  # (B, chunk, V)
        out = torch.zeros(B, self.vocab_size, dtype=logits.dtype, device=self.device)
        for b in range(B):
            last = int(seq_lengths[b]) - 1
            out[b] = logits[b, last]
        return out

    def decode(
        self,
        input_ids: torch.Tensor,
        kv: PagedKVCache,
        seq_ids: Sequence[int],
        ctx_lengths: Sequence[int],
    ) -> torch.Tensor:
        """Decode one freshly sampled token per row; return next-token logits ``(B, V)``."""
        self.forward_calls += 1
        B = input_ids.shape[0]
        hidden = self.embed_tokens(input_ids)          # (B, H)
        positions = torch.tensor(ctx_lengths, dtype=torch.long, device=self.device)
        cos = self._cos_sin[positions, 0].unsqueeze(1)  # (B, 1, d): broadcasts over heads
        sin = self._cos_sin[positions, 1].unsqueeze(1)
        for index, layer in enumerate(self.layers):
            hidden = layer.decode_step(
                hidden, kv, index, seq_ids, ctx_lengths, cos, sin
            )
        return self.lm_head(self.norm(hidden))

    def _prefill_layer(
        self,
        hidden,
        layer: PagedDecoderLayer,
        layer_index: int,
        kv: PagedKVCache,
        seq_ids: Sequence[int],
        seq_lengths: Sequence[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, chunk = hidden.shape[0], hidden.shape[1]
        x = layer.input_layernorm(hidden)
        q_raw = layer.q(x).view(B, chunk, layer.num_q_heads, layer.head_dim)
        k_raw = layer.k(x).view(B, chunk, layer.num_kv_heads, layer.head_dim)
        v_raw = layer.v(x).view(B, chunk, layer.num_kv_heads, layer.head_dim)
        # cos/sin (chunk, d) broadcast over (B, chunk, heads, d).
        q = apply_rotary(q_raw, sin[None, :, None, :], cos[None, :, None, :])
        k = apply_rotary(k_raw, sin[None, :, None, :], cos[None, :, None, :])
        out = paged_attention_prefill_chunk(
            q, k, v_raw, layer.num_q_heads, seq_lengths
        )  # (B, chunk, q_heads, d); padded positions zeroed
        for b in range(B):
            length = int(seq_lengths[b])
            for tpos in range(length):
                kv.write(layer_index, seq_ids[b], tpos, k[b, tpos], v_raw[b, tpos])
        attn_out = layer.o(out.reshape(B, chunk, -1))
        hidden = (hidden + attn_out)                    # out-proj of the *normed* input
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden