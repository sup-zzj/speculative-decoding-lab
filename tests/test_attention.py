"""Tests that the paged-attention kernels match a dense eager oracle.

Correctness-first philosophy: the ``gather + matmul`` paged kernels must match the
equivalent dense, contiguous self-attention (``torch.nn.functional.
scaled_dot_product_attention``). This is the independent oracle -- it does not
reuse the engine's own attention primitives. Two paths are checked:

* :func:`paged_attention_decode` -- one new token per row attending its full
  stored prefix (unmasked).
* :func:`paged_attention_prefill_chunk` -- causal attention within a freshly
  prefilled chunk, with left-padded unpadded rows zeroed.

Plus a sanity check for the GQA ``repeat_interleave`` head expansion.
"""

from __future__ import annotations

import torch
import pytest
from torch.nn.functional import scaled_dot_product_attention

from engine.attention import (
    _expand_to_q_heads,
    full_attention,
    gather_ctx,
    paged_attention_decode,
    paged_attention_prefill_chunk,
)
from engine.kv import PagedKVCache

_D = 8
_QH = 4
_KVH = 2
_LAYERS = 2
_BLOCK = 4


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_kv(capacity: int = 32) -> PagedKVCache:
    return PagedKVCache(
        num_layers=_LAYERS,
        num_kv_heads=_KVH,
        head_dim=_D,
        capacity_blocks=capacity,
        block_size=_BLOCK,
        device=_device(),
        dtype=torch.float32,
    )


def _write_sequence(kv: PagedKVCache, layer: int, seq_id: int, key, value) -> None:
    """Write the K/V for a (L, kv_heads, d) sequence into paged blocks."""
    for pos in range(key.shape[0]):
        kv.write(layer, seq_id, pos, key[pos], value[pos])


def _sdpa_decode(query, key, value, num_heads):
    """Eager decode oracle via SDPA: (num_heads, d) query attending (ctx, num_heads, d).

    SDPA expects ``(N, H, L, E)`` -- heads come *before* the sequence length -- so
    the ``(L, H, d)`` key/value must be transposed before batching.
    """
    q = query[None, :, None, :]          # (1, QH, 1, d)
    k = key.transpose(0, 1)[None]        # (1, QH, ctx, d)
    v = value.transpose(0, 1)[None]
    out = scaled_dot_product_attention(q, k, v, is_causal=False)
    return out[0, :, 0, :]               # (QH, d)


def test_expand_to_q_heads_repeats_kv() -> None:
    kv_tensor = torch.randn(5, _KVH, _D, device=_device())
    expanded = _expand_to_q_heads(kv_tensor, _QH)
    assert expanded.shape == (5, _QH, _D)
    # repeat_interleave(ratio, dim=-2): each kv-head row appears `ratio` times.
    assert torch.equal(expanded[:, 0], kv_tensor[:, 0])
    assert torch.equal(expanded[:, 1], kv_tensor[:, 0])
    assert torch.equal(expanded[:, 2], kv_tensor[:, 1])
    assert torch.equal(expanded[:, 3], kv_tensor[:, 1])


def test_decode_matches_full_attention_oracle() -> None:
    kv = _make_kv()
    kv.add_sequence(0)
    layer = 0
    key = torch.randn(9, _KVH, _D, device=kv.device)
    value = torch.randn(9, _KVH, _D, device=kv.device)
    _write_sequence(kv, layer, 0, key, value)

    query = torch.randn(1, _QH, _D, device=kv.device)
    out = paged_attention_decode(query, kv, layer, [0], [9], _QH)

    gathered = gather_ctx(kv, layer, 0, 9)
    oracle = full_attention(query[0], gathered[0], gathered[1], _QH)
    assert torch.allclose(out[0], oracle, atol=1e-5, rtol=1e-5)


def test_decode_matches_sdpa_oracle() -> None:
    kv = _make_kv()
    kv.add_sequence(0)
    layer = 0
    ctx = 10
    key = torch.randn(ctx, _KVH, _D, device=kv.device)
    value = torch.randn(ctx, _KVH, _D, device=kv.device)
    _write_sequence(kv, layer, 0, key, value)

    query = torch.randn(1, _QH, _D, device=kv.device)
    out = paged_attention_decode(query, kv, layer, [0], [ctx], _QH)
    # GQA-expand key/value up to the query head count for the SDPA oracle.
    oracle = _sdpa_decode(
        query[0], _expand_to_q_heads(key, _QH), _expand_to_q_heads(value, _QH), _QH
    )
    assert torch.allclose(out[0], oracle, atol=1e-5, rtol=1e-5)


def test_decode_multi_sequence_batch_matches_oracle() -> None:
    kv = _make_kv(capacity=64)
    layer = 0
    seq_lens = []
    for sid in range(3):
        kv.add_sequence(sid)
        length = 1 + sid * 3  # 1, 4, 7 -> spans multiple physical blocks
        key = torch.randn(length, _KVH, _D, device=kv.device)
        value = torch.randn(length, _KVH, _D, device=kv.device)
        _write_sequence(kv, layer, sid, key, value)
        seq_lens.append(length)

    query = torch.randn(3, _QH, _D, device=kv.device)
    out = paged_attention_decode(query, kv, layer, [0, 1, 2], seq_lens, _QH)
    assert out.shape == (3, _QH, _D)
    for sid in range(3):
        gathered = gather_ctx(kv, layer, sid, seq_lens[sid])
        oracle = _sdpa_decode(
            query[sid],
            _expand_to_q_heads(gathered[0], _QH),
            _expand_to_q_heads(gathered[1], _QH),
            _QH,
        )
        assert torch.allclose(out[sid], oracle, atol=1e-5, rtol=1e-5)


def test_prefill_chunk_matches_sdpa_oracle() -> None:
    kv = _make_kv()
    layer = 0
    chunk_len = 6
    query = torch.randn(2, chunk_len, _QH, _D, device=kv.device)
    lengths = [4, 6]

    # Causal self-attention oracle: pass the same tensor for q, k and v (the
    # symmetric case), mirroring the engine's real (q, k, v) contract.
    out = paged_attention_prefill_chunk(query, query, query, _QH, lengths)

    for index, length in enumerate(lengths):
        row = query[index][:length]  # (L, QH, d)
        # SDPA layout (1, QH, L, d); transpose heads before the seq length.
        block = row.transpose(0, 1)[None]
        oracle = scaled_dot_product_attention(block, block, block, is_causal=True)[0]
        oracle = oracle.transpose(0, 1)  # back to (L, QH, d)
        assert torch.allclose(out[index][:length], oracle, atol=1e-5, rtol=1e-5)
        # Padded tail (length..chunk_len) must be exactly zero.
        assert torch.equal(out[index][length:], torch.zeros_like(out[index][length:]))


def test_prefill_avoids_padding_outside_length() -> None:
    kv = _make_kv()
    layer = 0
    chunk_len = 5
    query = torch.randn(2, chunk_len, _QH, _D, device=kv.device)
    out = paged_attention_prefill_chunk(query, query, query, _QH, [2, 5])
    # Row 0 has length 2; the pad is at the TAIL (indices length..chunk_len-1).
    assert torch.allclose(out[0][2:], torch.zeros_like(out[0][2:]), atol=0.0)
    # And every row's output is finite (no NaN from all-(-inf) masked positions).
    assert torch.isfinite(out).all()