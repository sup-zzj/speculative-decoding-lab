"""Paged attention kernels (gather-based, correctness first).

The engine stores every sequence's K/V in fixed-size physical blocks. To attend,
we *gather* the physical blocks of a sequence into a dense ``(ctx_len, kv_heads,
head_dim)`` tensor and run a standard masked softmax attention against the query.
This is the "gather + matmul" formulation of PagedAttention (contrast with
``torch.nn.functional.scaled_dot_product_attention`` on a contiguous cache); it is
deliberately simple and directly verifiable against an eager oracle.

Two entry points cover both scheduling phases:

* :func:`paged_attention_decode` -- one freshly fed query token per row. The token
  attends *every* already-stored position (its own K/V is written afterwards), so
  this is an unmasked attention over the gathered prefix.
* :func:`paged_attention_prefill_chunk` -- a whole prompt chunk per row with
  causal masking *within* the chunk and no external prefix yet (``query_start=0``).

Grouped-query attention (GQA) is handled by expanding ``kv_heads`` to the number
of query heads with ``repeat_interleave``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

from .kv import PagedKVCache


def _expand_to_q_heads(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Broadcast ``(.., kv_heads, d)`` to ``(.., num_q_heads, d)`` for GQA."""
    kv_heads = kv.shape[-2]
    ratio = num_q_heads // kv_heads
    if ratio == 1:
        return kv
    return kv.repeat_interleave(ratio, dim=-2)


def gather_ctx(kv: PagedKVCache, layer: int, seq_id: int, length: int) -> torch.Tensor:
    """Return the stored K/V prefix of one sequence as ``(2, length, kv_heads, d)``.

    Row 0 is K, row 1 is V. Physical blocks are concatenated in logical order;
    only ``length`` cells are taken, so freed/stale blocks never leak in.
    """
    if length <= 0:
        return torch.zeros(
            (2, 0, kv.num_kv_heads, kv.head_dim), dtype=kv.dtype, device=kv.device
        )
    table = kv.block_tables[seq_id]
    block_size = kv.block_size
    n_blocks = (length + block_size - 1) // block_size
    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    for bi in range(n_blocks):
        phys = table[bi]
        take = min(block_size, length - bi * block_size)
        k_parts.append(kv.key_map(layer, [phys])[0, :take])
        v_parts.append(kv.value_map(layer, [phys])[0, :take])
    key = torch.cat(k_parts, dim=0)
    value = torch.cat(v_parts, dim=0)
    return torch.stack((key, value), dim=0)


def full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """Unmasked scaled dot-product attention over a full key/value prefix.

    ``query`` shape ``(num_q_heads, d)``; ``key``/``value`` shape
    ``(ctx_len, kv_heads, d)`` (GQA-expanded internally). Returns
    ``(num_q_heads, d)``. Used by the decode path where the new token sees all
    stored positions.

    Attention is computed with :func:`torch.nn.functional.scaled_dot_product_attention`
    on the *gathered* dense K/V. This is the same CUDA kernel the reference
    ``transformers`` decoder runs, so on real weights the paged engine reproduces
    the autoregressive token stream bit-for-bit; an fp16 einsum softmax would
    otherwise drift and diverge token 0 within a handful of decode steps.
    """
    d = query.shape[-1]
    key = _expand_to_q_heads(key, num_q_heads)
    value = _expand_to_q_heads(value, num_q_heads)
    # Restack to SDPA layout ``(B=1, heads, seq, head_dim)``.
    q4 = query.unsqueeze(0).unsqueeze(2)      # (1, num_q_heads, 1, d)
    k4 = key.transpose(0, 1).unsqueeze(0)     # (1, num_q_heads, ctx, d)
    v4 = value.transpose(0, 1).unsqueeze(0)   # (1, num_q_heads, ctx, d)
    attn = F.scaled_dot_product_attention(q4, k4, v4)
    return attn.squeeze(0).squeeze(1)          # (num_q_heads, d)


def paged_attention_decode(
    query_2d: torch.Tensor,
    kv: PagedKVCache,
    layer: int,
    seq_ids: Sequence[int],
    seq_token_lengths: Sequence[int],
    num_q_heads: int,
) -> torch.Tensor:
    """Attend one new query token per row against its full stored prefix.

    ``query_2d`` is ``(B, num_q_heads, d)`` (one projected+RoPE-d query per row).
    Each row ``i`` gathers ``seq_ids[i]`` up to ``seq_token_lengths[i]`` stored
    positions and runs an unmasked attention (the classic paged decode step).
    Returns ``(B, num_q_heads, d)``.
    """
    outputs: List[torch.Tensor] = []
    for index in range(query_2d.shape[0]):
        query = query_2d[index]
        length = int(seq_token_lengths[index])
        key_value = gather_ctx(kv, layer, seq_ids[index], length)
        key, value = key_value[0], key_value[1]
        outputs.append(full_attention(query, key, value, num_q_heads))
    return torch.stack(outputs, dim=0)


def paged_attention_prefill_chunk(
    query_3d: torch.Tensor,
    key_3d: torch.Tensor,
    value_3d: torch.Tensor,
    num_q_heads: int,
    chunk_query_lengths: Sequence[int],
) -> torch.Tensor:
    """Causal attention over a freshly prefilled prompt chunk (``query_start=0``).

    ``query_3d`` is ``(B, chunk_len, num_q_heads, d)``; ``key_3d``
    ``(B, chunk_len, num_kv_heads, d)`` (already RoPE-rotated); ``value_3d``
    ``(B, chunk_len, num_kv_heads, d)`` (raw, RoPE is never applied to V). Row
    ``i``'s position ``j`` attends ``0..j`` *within the same chunk*. The real
    projected K/V are required here -- the function must not re-derive them from
    the query, or the prefill attention is wrong. ``chunk_query_lengths[i]`` is
    the unpadded length of row ``i`` (rows are tail-padded to ``chunk_len``);
    padded positions are zeroed and discarded by the caller. Returns
    ``(B, chunk_len, num_q_heads, d)``.
    """
    B = query_3d.shape[0]
    chunk_len = query_3d.shape[1]
    outputs: List[torch.Tensor] = []
    for index in range(B):
        length = int(chunk_query_lengths[index])
        q = query_3d[index][:length]                                # (L, qh, d)
        k = _expand_to_q_heads(key_3d[index][:length], num_q_heads)  # (L, qh, d)
        v = _expand_to_q_heads(value_3d[index][:length], num_q_heads)  # (L, qh, d)
        # Causal SDPA over the valid slice, matching the reference kernel so an
        # fp16 prefill is bit-exact (see :func:`full_attention`).
        q4 = q.unsqueeze(0).transpose(1, 2)   # (1, qh, L, d)
        k4 = k.unsqueeze(0).transpose(1, 2)   # (1, qh, L, d)
        v4 = v.unsqueeze(0).transpose(1, 2)   # (1, qh, L, d)
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0]
        out = out.transpose(0, 1)             # (L, qh, d)
        # Right-pad so the caller can index the real content at logical 0..L-1
        # (e.g. ``logits[b, length - 1]`` for the last token). Padded slots are
        # zeroed and never written to KV.
        pad_len = chunk_len - length
        if pad_len > 0:
            pad = torch.zeros(
                (pad_len, q.shape[-2], q.shape[-1]),
                dtype=q.dtype,
                device=q.device,
            )
            out = torch.cat((out, pad), dim=0)
        outputs.append(out)
    return torch.stack(outputs, dim=0)  # B, chunk_len, q_heads, d


def paged_attention_extend(
    query_3d: torch.Tensor,
    kv: PagedKVCache,
    layer: int,
    seq_ids: Sequence[int],
    start_positions: Sequence[int],
    num_q_heads: int,
) -> torch.Tensor:
    """Attend a chunk of fresh tokens against an existing prefix plus prior rows.

    ``query_3d`` is ``(B, chunk, num_q_heads, d)`` (already projected + RoPE'd).
    Each sequence already had its ``prefix + new chunk`` K/V written to ``kv``, so
    ``kv.seq_lengths[sid] == start_positions[b] + chunk``. ``start_positions[b]`` is
    the logical position where sequence ``b``'s chunk begins (its committed prefix
    length). Row ``j`` of sequence ``b`` therefore sits at logical position
    ``start_positions[b] + j`` and attends every position ``0 .. start_positions[b] + j``
    (the full external prefix plus the chunk rows before ``j``).

    Returns ``(B, chunk, num_q_heads, d)``. Computed as one masked
    :func:`torch.nn.functional.scaled_dot_product_attention` per sample with a boolean
    causal mask (``mask[j, i] = i <= start+j``), so it stays bit-consistent with the
    other gather-based kernels.
    """
    B, chunk, _, _d = query_3d.shape
    outputs: List[torch.Tensor] = []
    for b in range(B):
        start = int(start_positions[b])
        length = start + chunk
        key_value = gather_ctx(kv, layer, seq_ids[b], length)
        key = _expand_to_q_heads(key_value[0], num_q_heads)    # (L, qh, d)
        value = _expand_to_q_heads(key_value[1], num_q_heads)
        # Causal mask: query row ``j`` (absolute position ``start+j``) attends key
        # rows ``0 .. start+j`` inclusive. Building it by row keeps the mask cheap
        # and correct even when ``length`` (and so the matrix) is small.
        mask = torch.zeros(
            (chunk, length), dtype=torch.bool, device=query_3d.device
        )
        for j in range(chunk):
            mask[j, : start + j + 1] = True
        q4 = query_3d[b].transpose(0, 1).unsqueeze(0)   # (1, qh, chunk, d)
        k4 = key.transpose(0, 1).unsqueeze(0)           # (1, qh, L, d)
        v4 = value.transpose(0, 1).unsqueeze(0)         # (1, qh, L, d)
        attn = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask.unsqueeze(0)
        )
        outputs.append(attn[0].transpose(0, 1))          # (chunk, qh, d)
    return torch.stack(outputs, dim=0)  # (B, chunk, qh, d)