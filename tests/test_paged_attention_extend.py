"""Tests for ``paged_attention_extend`` against a per-position prefix oracle.

A fresh prompt is prefilled into one decoder layer's KV, then a chunk of random
tokens is appended. ``paged_attention_extend(q3d, ...)`` row ``j`` must equal
``full_attention(q3d[b, j], prefix keys 0..start+j+1)`` -- the same dense prefix
attention the kernel is meant to generalise -- for every position ``j``.
"""

from __future__ import annotations

import torch
import pytest

from engine.attention import (
    full_attention,
    gather_ctx,
    paged_attention_extend,
)
from engine.config import EngineConfig
from engine.kv import PagedKVCache
from engine.model import (
    ToyConfig,
    ToyPagedLM,
    apply_rotary,
)
from engine.scheduler import build_kv

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def model() -> ToyPagedLM:
    return ToyPagedLM(ToyConfig(seed=0), _DEVICE)


def _append_chunk_layer0(model: ToyPagedLM, kv: PagedKVCache, start: int, chunk_ids: torch.Tensor):
    """Project + RoPE the chunk, write its K/V into layer 0, return the query 3D."""
    layer = model.layers[0]
    hidden = model.embed_tokens(chunk_ids)  # (1, chunk, H)
    B, chunk, H = hidden.shape
    x = layer.input_layernorm(hidden)
    qh, kh, d = layer.num_q_heads, layer.num_kv_heads, layer.head_dim
    q_raw = layer.q(x).view(B, chunk, qh, d)
    k_raw = layer.k(x).view(B, chunk, kh, d)
    v_raw = layer.v(x).view(B, chunk, kh, d)
    positions = torch.arange(start, start + chunk, dtype=torch.long, device=model.device)
    cos = model._cos_sin[positions, 0][None, :, None, :]   # (1, chunk, 1, d)
    sin = model._cos_sin[positions, 1][None, :, None, :]
    q = apply_rotary(q_raw, sin, cos)
    k = apply_rotary(k_raw, sin, cos)
    for j in range(chunk):
        kv.write(0, 0, start + j, k[0, j], v_raw[0, j])
    return q  # (1, chunk, qh, d)


def test_extend_matches_full_attention_oracle() -> None:
    model = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=64)
    kv = build_kv(model, cfg, _DEVICE)

    torch.manual_seed(1)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 7), device=_DEVICE)
    kv.add_sequence(0)
    model.prefill(prompt, kv, [0], [int(prompt.shape[1])])  # writes prompt K/V -> seq_len 7

    start = int(kv.seq_lengths[0])
    chunk = 5
    chunk_ids = torch.randint(0, model.cfg.vocab_size, (1, chunk), device=_DEVICE)
    q3d = _append_chunk_layer0(model, kv, start, chunk_ids)  # (1, chunk, qh, d)

    out = paged_attention_extend(q3d, kv, 0, [0], [start], model.num_q_heads)
    assert out.shape == (1, chunk, model.num_q_heads, model.head_dim)

    for j in range(chunk):
        kv_pair = gather_ctx(kv, 0, 0, start + j + 1)
        oracle = full_attention(q3d[0, j], kv_pair[0], kv_pair[1], model.num_q_heads)
        assert torch.allclose(out[0, j], oracle, atol=1e-5, rtol=1e-5)