"""Real-model adapter: run Qwen2.5 weights through the paged engine.

This adapter builds a decoder stack whose parameter *names* mirror ``transformers``
exactly (so ``load_state_dict`` transfers pretrained weights in one call), but whose
forward path routes attention through :mod:`engine.attention.paged_attention_*` over a
:class:`PagedKVCache` -- i.e. it is a from-scratch, PagedAttention serving forward on
real weights, with no reliance on ``transformers`` caching internals.

RoPE (theta from the config, hand-rolled), RMSNorm, GQA and the SwiGLU MLP are all
reimplemented here and share the exact primitives the toy model uses, so the
correctness harness can validate the whole stack token-for-token against the native
``transformers`` path on the *same* weights.
"""

from __future__ import annotations

import os
import torch
from torch import nn
from typing import Any, List, Optional, Sequence

from .attention import paged_attention_decode, paged_attention_prefill_chunk
from .kv import PagedKVCache
from .model import RMSNorm, apply_rotary, make_rotary_cache
from .utils import setup_logger

logger = setup_logger(__name__)


class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        theta: float,
        max_len: int,
        device: torch.device,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # ``transformers`` Qwen2 layers gate q/k/v projections on ``attention_bias``
        # (default True). Matching it is essential: a missing bias silently shifts
        # every q/k/v token, which on real weights destroys prefill logits.
        self.q_proj = nn.Linear(hidden, num_q_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.rotary_theta = theta
        # Store the hand-rolled RoPE table the same way the toy model does.
        self.register_buffer(
            "_cos_sin", make_rotary_cache(head_dim, theta, max_len, device), persistent=False
        )


class QwenMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    """A Qwen2 decoder layer whose names match ``transformers`` for state_dict."""

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate: int,
        theta: float,
        max_len: int,
        eps: float,
        device: torch.device,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = QwenAttention(
            hidden, num_q_heads, num_kv_heads, head_dim, theta, max_len, device,
            attention_bias=attention_bias,
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = QwenMLP(hidden, intermediate)

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
        B = hidden.shape[0]
        attn = self.self_attn
        qh, kh, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
        residual = hidden
        x = self.input_layernorm(hidden)
        q_raw = attn.q_proj(x).view(B, qh, d)
        k_raw = attn.k_proj(x).view(B, kh, d)
        v_raw = attn.v_proj(x).view(B, kh, d)
        q = apply_rotary(q_raw, sin, cos)
        k = apply_rotary(k_raw, sin, cos)
        out = paged_attention_decode(q, kv, layer_index, seq_ids, ctx_lengths, qh)
        # Only K is RoPE-rotated; store the new token's own K/V at position
        # ctx_lengths[b] FIRST (see model.py fix), then attend every position
        # up to and including itself so causal attention matches the reference.
        for b in range(B):
            kv.write(layer_index, seq_ids[b], int(ctx_lengths[b]), k[b], v_raw[b])
        out = paged_attention_decode(
            q, kv, layer_index, seq_ids, [int(c) + 1 for c in ctx_lengths], qh
        )
        hidden = residual + attn.o_proj(out.view(B, -1))
        residual = hidden
        hidden = residual + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def prefill_layer(
        self,
        hidden: torch.Tensor,
        layer_index: int,
        kv: PagedKVCache,
        seq_ids: Sequence[int],
        seq_lengths: Sequence[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, chunk = hidden.shape[0], hidden.shape[1]
        attn = self.self_attn
        qh, kh, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
        x = self.input_layernorm(hidden)
        q_raw = attn.q_proj(x).view(B, chunk, qh, d)
        k_raw = attn.k_proj(x).view(B, chunk, kh, d)
        v_raw = attn.v_proj(x).view(B, chunk, kh, d)
        q = apply_rotary(q_raw, sin[None, :, None, :], cos[None, :, None, :])
        k = apply_rotary(k_raw, sin[None, :, None, :], cos[None, :, None, :])
        out = paged_attention_prefill_chunk(q, k, v_raw, qh, seq_lengths)
        for b in range(B):
            length = int(seq_lengths[b])
            for tpos in range(length):
                kv.write(layer_index, seq_ids[b], tpos, k[b, tpos], v_raw[b, tpos])
        attn_out = attn.o_proj(out.reshape(B, chunk, -1))
        hidden = hidden + attn_out
        residual = hidden
        hidden = residual + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class QwenPagedModel(nn.Module):
    """A Qwen2.5 causal LM whose forward consumes ``PagedKVCache``."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.num_layers = int(config.num_hidden_layers)
        self.num_q_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_q_heads))
        self._head_dim = int(
            getattr(config, "head_dim", 0)
            or ( (config.hidden_size // self.num_q_heads) )
        )
        self._theta = float(getattr(config, "rope_theta", 10000.0))
        self._eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self._max_len = int(
            getattr(config, "max_position_embeddings", 2048)
        )
        self._attn_bias = bool(getattr(config, "attention_bias", True))
        self.vocab_size = int(config.vocab_size)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

        self.embed_tokens = nn.Embedding(self.vocab_size, int(config.hidden_size))
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    int(config.hidden_size),
                    self.num_q_heads,
                    self.num_kv_heads,
                    self._head_dim,
                    int(config.intermediate_size),
                    self._theta,
                    self._max_len,
                    self._eps,
                    device,
                    self._attn_bias,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = RMSNorm(int(config.hidden_size), self._eps)
        self.lm_head = nn.Linear(int(config.hidden_size), self.vocab_size, bias=False)
        if self._tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.forward_calls = 0

    # ----------------------------------------------------------- properties
    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def reset_counters(self) -> None:
        self.forward_calls = 0

    # ------------------------------------------------------------ rotary fetch
    def _rope_for(self, positions: torch.Tensor) -> torch.Tensor:
        table = self.layers[0].self_attn._cos_sin
        return table[positions]

    # ------------------------------------------------------------ paged api
    def prefill(
        self,
        prompt_2d: torch.Tensor,
        kv: PagedKVCache,
        seq_ids: Sequence[int],
        seq_lengths: Sequence[int],
    ) -> torch.Tensor:
        self.forward_calls += 1
        B, chunk = prompt_2d.shape
        hidden = self.embed_tokens(prompt_2d)
        positions = torch.arange(chunk, dtype=torch.long, device=self.device)
        cs = self._rope_for(positions)
        cos, sin = cs[:, 0], cs[:, 1]
        for index, layer in enumerate(self.layers):
            hidden = layer.prefill_layer(
                hidden, index, kv, seq_ids, seq_lengths, cos, sin
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
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
        self.forward_calls += 1
        B = input_ids.shape[0]
        hidden = self.embed_tokens(input_ids)
        positions = torch.tensor(ctx_lengths, dtype=torch.long, device=self.device)
        cs = self._rope_for(positions)
        cos, sin = cs[:, 0].unsqueeze(1), cs[:, 1].unsqueeze(1)
        for index, layer in enumerate(self.layers):
            hidden = layer.decode_step(
                hidden, kv, index, seq_ids, ctx_lengths, cos, sin
            )
        return self.lm_head(self.norm(hidden))


# ------------------------------------------------------------- loader helpers
def resolve_local(model_id: str, cache_dir: str) -> str:
    """Resolve ``models/...`` or ``cache_dir/<basename>`` to a local checkpoint dir."""
    candidates = [
        model_id,
        os.path.join(cache_dir, model_id),
        os.path.join(cache_dir, os.path.basename(model_id)),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and any(
            name.startswith("model") and name.endswith((".safetensors", ".bin"))
            for name in os.listdir(candidate)
        ):
            return candidate
    return model_id


def build_paged_model(
    model_id: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> "QwenPagedModel":
    """Load real Qwen2.5 weights into the paged engine, then free the transformers copy."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    source = resolve_local(model_id, cache_dir)
    config = AutoConfig.from_pretrained(source, cache_dir=cache_dir)
    native = AutoModelForCausalLM.from_pretrained(
        source, cache_dir=cache_dir, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = QwenPagedModel(config, device)
    model = model.to(device=device, dtype=dtype)
    model.tokenizer = tokenizer
    # ``transformers`` prefixes every ``Qwen2Model`` weight with ``model.``
    # (e.g. ``model.layers.0.self_attn.q_proj.weight``) but ``lm_head`` is a
    # direct child (``lm_head.weight``). Strip the prefix so the keys line up
    # with our ``QwenPagedModel`` names before transferring the weights.
    native_state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in native.state_dict().items()
    }
    missing, unexpected = model.load_state_dict(native_state, strict=False)
    if missing:
        raise RuntimeError("missing weights: {}".format(missing))
    if unexpected:
        logger.warning("ignored %d unexpected weight keys (e.g. rotary buffers)", len(unexpected))
    del native
    torch.cuda.empty_cache() if device.type == "cuda" else None
    model.eval()
    logger.info(
        "loaded %s -> paged engine (%d layers, %d Q / %d KV heads, d=%d, %s)",
        source, model.num_layers, model.num_q_heads, model.num_kv_heads,
        model.head_dim, dtype,
    )
    return model