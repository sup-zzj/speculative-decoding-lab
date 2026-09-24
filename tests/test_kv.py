"""Unit tests for the paged KV cache and block allocator.

Covers the two invariants the whole engine relies on:

1. **on-demand block growth** -- ``write`` allocates physical blocks lazily and
   advances ``seq_lengths``;
2. **per-sample rollback** -- ``truncate`` rewinds a sequence to an earlier
   length, frees the surplus blocks, and the returned blocks are reusable.

Both are the exact primitives PagedAttention serving (and later the speculative
decoding Phase 3) need.
"""

from __future__ import annotations

import torch
import pytest

from engine.kv import BlockAllocator, BlockExhaustedError, PagedKVCache
from engine.attention import gather_ctx


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kv(block_size: int = 4, capacity: int = 8) -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        capacity_blocks=capacity,
        block_size=block_size,
        device=_device(),
        dtype=torch.float32,
    )


def _vec(head_dim: int = 8, fill: float = 1.0) -> torch.Tensor:
    return torch.full((2, head_dim), fill, dtype=torch.float32, device=_device())


# ------------------------------------------------------------------ allocator
def test_allocator_alloc_free_roundtrip() -> None:
    allocator = BlockAllocator(capacity=4, device=_device(), dtype=torch.float32)
    blocks = [allocator.alloc() for _ in range(4)]
    assert sorted(blocks) == [0, 1, 2, 3]
    assert allocator.num_allocated == 4
    allocator.free(blocks[0])
    allocator.free(blocks[1])
    assert allocator.num_allocated == 2
    blocks.extend([allocator.alloc(), allocator.alloc()])
    assert allocator.num_allocated == 4


def test_allocator_exhaustion_raises() -> None:
    allocator = BlockAllocator(capacity=2, device=_device(), dtype=torch.float32)
    allocator.alloc()
    allocator.alloc()
    with pytest.raises(BlockExhaustedError):
        allocator.alloc()
    # Free one and it works again.
    allocator.free(0)
    assert allocator.alloc() == 0


def test_allocator_free_twice_is_idempotent() -> None:
    allocator = BlockAllocator(capacity=3, device=_device(), dtype=torch.float32)
    block = allocator.alloc()
    allocator.free(block)
    allocator.free(block)  # double free must not corrupt the free list
    assert allocator.num_allocated == 0


# ------------------------------------------------------------------ paged cache
def test_write_allocates_blocks_on_demand() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    kv.write(0, 0, 0, _vec(), _vec())
    assert kv.seq_lengths[0] == 1
    assert kv.num_allocated_blocks == 1
    # Writing a third token reuses the same block (positions < block_size).
    kv.write(0, 0, 2, _vec(), _vec())
    assert kv.num_allocated_blocks == 1
    assert kv.seq_lengths[0] == 3
    # Position 4 rolls into a second block.
    kv.write(0, 0, 4, _vec(), _vec())
    assert kv.num_allocated_blocks == 2
    assert len(kv.block_tables[0]) == 2


def test_write_preserves_order_and_extends_contiguously() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    kv.write(0, 0, 3, _vec(fill=3.0), _vec(fill=3.0))
    # An out-of-order write via token_pos extends length to cover both.
    assert kv.seq_lengths[0] == 4
    assert kv.logical_tokens == 4


def test_gather_concatenates_blocks_in_order() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    for pos in range(9):
        kv.write(0, 0, pos, _vec(fill=float(pos)), _vec(fill=-float(pos)))
    stacked = gather_ctx(kv, 0, 0, 9)
    keys = stacked[0]  # (9, kv_heads, d)
    assert keys.shape == (9, 2, 8)
    # The gathered key at logical position p should equal the written fill.
    assert torch.allclose(keys[3], torch.full((2, 8), 3.0, device=kv.device))


def test_truncate_rollback_frees_blocks() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    kv.add_sequence(1)
    for pos in range(10):
        kv.write(0, 0, pos, _vec(fill=1.0), _vec())
    before = kv.num_allocated_blocks
    assert before >= 3
    kv.truncate(0, 3)  # one block can hold 4 positions, so 1 block remains
    assert kv.seq_lengths[0] == 3
    assert len(kv.block_tables[0]) == 1
    assert kv.num_allocated_blocks < before
    # The freed block can be handed to another sequence via the same pool.
    kv.write(0, 1, 4, _vec(), _vec())
    assert kv.seq_lengths[1] == 5
    assert len(kv.block_tables[1]) == 2


def test_truncate_rewound_sequence_writes_identically() -> None:
    """Rewind + re-extend reproduces the exact original values (rollback invariant)."""
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    for pos in range(8):
        kv.write(0, 0, pos, _vec(fill=float(pos) + 0.5), _vec())
    # Rewind to 2, then re-write positions 2..7 (same logical mapping).
    kv.truncate(0, 2)
    for pos in range(2, 8):
        kv.write(0, 0, pos, _vec(fill=float(pos) + 0.5), _vec())
    stacked = gather_ctx(kv, 0, 0, 8)
    assert torch.allclose(
        stacked[0][5], torch.full((2, 8), 5.5, device=kv.device)
    )


def test_clear_frees_all_blocks() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    for pos in range(10):
        kv.write(0, 0, pos, _vec(), _vec())
    used = kv.num_allocated_blocks
    assert used > 0
    kv.clear(0)
    assert kv.num_allocated_blocks == 0
    assert kv.seq_lengths[0] == 0
    assert kv.block_tables[0] == []


def test_reset_returns_cache_to_virgin_state() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    kv.add_sequence(1)
    for pos in range(6):
        kv.write(0, 0, pos, _vec(), _vec())
    kv.write(0, 1, 1, _vec(), _vec())
    assert kv.num_sequences == 2
    assert kv.num_allocated_blocks > 0
    kv.reset()
    assert kv.num_sequences == 0
    assert kv.num_allocated_blocks == 0
    # The tensors are reused (same object), not reallocated.
    assert kv.num_allocated_blocks == 0


def test_block_utilization_metric() -> None:
    kv = _kv(block_size=4)
    kv.add_sequence(0)
    kv.write(0, 0, 0, _vec(), _vec())
    # 1 token in 1 block of 4 slots.
    assert kv.block_utilization == pytest.approx(0.25)
    kv.write(0, 0, 1, _vec(), _vec())
    assert kv.block_utilization == pytest.approx(0.5)