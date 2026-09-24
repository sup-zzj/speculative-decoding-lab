"""Paged KV cache and physical block allocator.

The whole Phase-2 engine is built on a single primitive: **each sequence's
key/value cache lives in fixed-size physical blocks, mapped by a per-sequence
block table**. Because there is no shared batch dimension like
``transformers.DynamicCache``, sequences of *different* lengths coexist freely
and each one can grow, be dropped, or be *rewound* independently -- exactly the
per-sample rollback that heterogeneous batching (and, later, speculative
decoding) needs.

Layout
------
``K`` and ``V`` are each a tensor of shape

    (num_layers, capacity_blocks, block_size, num_kv_heads, head_dim)

A block is ``block_size`` consecutive *logical positions* of one layer. Writing
the K/V of token at logical position ``p`` of sequence ``s`` goes to

    phys    = block_tables[s][p // block_size]
    offset  = p % block_size
    K[k, phys, offset, :, :] = kvec

``truncate`` rewinds a sequence to ``new_len`` logical positions and returns the
freed physical blocks to the allocator; because every token position is
recomputed and rewritten when the sequence is re-extended, a rewound sequence
decodes identically to one that never extended (the rollback invariant).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .utils import setup_logger

logger = setup_logger(__name__)


class BlockExhaustedError(RuntimeError):
    """Raised when the physical block pool is exhausted by an allocation."""


class BlockAllocator:
    """Free-list of physical block ids shared by every sequence."""

    def __init__(self, capacity: int, device: torch.device, dtype: torch.dtype) -> None:
        if capacity < 1:
            raise ValueError("capacity_blocks must be >= 1")
        self.capacity = capacity
        self._free: List[int] = list(range(capacity))
        self._allocated: int = 0
        self._device = device
        self._dtype = dtype

    @property
    def num_allocated(self) -> int:
        return self._allocated

    def alloc(self) -> int:
        if not self._free:
            raise BlockExhaustedError(
                "paged KV block pool exhausted ({} blocks; raise max_num_blocks)".format(
                    self.capacity
                )
            )
        block = self._free.pop()
        self._allocated += 1
        return block

    def free(self, block: int) -> None:
        if block in self._free:
            return
        self._free.append(block)
        self._allocated -= 1


class PagedKVCache:
    """Physical KV tensors plus the per-sequence metadata that maps to them."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.allocator = BlockAllocator(capacity_blocks, device, dtype)
        shape = (num_layers, capacity_blocks, block_size, num_kv_heads, head_dim)
        self._k = torch.zeros(shape, dtype=dtype, device=device)
        self._v = torch.zeros(shape, dtype=dtype, device=device)
        #: ``block_tables[seq_id]`` is the list of physical blocks, in logical order.
        self.block_tables: List[List[int]] = []
        #: ``seq_lengths[seq_id]`` is the number of valid logical positions.
        self.seq_lengths: List[int] = []

    # ------------------------------------------------------------- sequences
    def add_sequence(self, seq_id: int) -> None:
        """Register an empty sequence (no blocks, zero length)."""
        while len(self.block_tables) <= seq_id:
            self.block_tables.append([])
            self.seq_lengths.append(0)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lengths)

    # ---------------------------------------------------------- block helpers
    def _ensure_blocks_up_to(self, seq_id: int, max_block_index: int) -> None:
        table = self.block_tables[seq_id]
        while len(table) <= max_block_index:
            table.append(self.allocator.alloc())

    # ------------------------------------------------------------ write path
    def write(
        self,
        layer: int,
        seq_id: int,
        token_pos: int,
        kvec: torch.Tensor,
        vvec: torch.Tensor,
    ) -> None:
        """Store the K/V vectors of one logical token position.

        ``kvec``/``vvec`` have shape ``(num_kv_heads, head_dim)``. The block that
        owns ``token_pos`` is allocated on demand; re-writing an already-occupied
        position simply overwrites it, which is what makes rollback exact.
        """
        block_index = token_pos // self.block_size
        offset = token_pos % self.block_size
        self._ensure_blocks_up_to(seq_id, block_index)
        phys = self.block_tables[seq_id][block_index]
        self._k[layer, phys, offset] = kvec
        self._v[layer, phys, offset] = vvec
        # A write by construction extends the valid length to cover the position.
        if self.seq_lengths[seq_id] <= token_pos:
            self.seq_lengths[seq_id] = token_pos + 1

    # ------------------------------------------------------------ rollback
    def truncate(self, seq_id: int, new_len: int) -> None:
        """Rewind sequence ``seq_id`` to ``new_len`` logical positions.

        Any physical block past the new length is freed. Positions still below
        ``new_len`` keep their blocks and (stale but unreachable) values -- they
        are never read beyond ``seq_lengths``, and will be overwritten if the
        sequence is re-extended.
        """
        blocks_needed = 0 if new_len <= 0 else (new_len + self.block_size - 1) // self.block_size
        table = self.block_tables[seq_id]
        while len(table) > blocks_needed:
            self.allocator.free(table.pop())
        self.seq_lengths[seq_id] = max(0, new_len)

    def clear(self, seq_id: int) -> None:
        """Drop a finished sequence entirely, freeing all its blocks."""
        table = self.block_tables[seq_id] if seq_id < len(self.block_tables) else []
        for phys in table:
            self.allocator.free(phys)
        # Remove the metadata row so a later sequence can reuse the slot.
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def reset(self) -> None:
        """Return the cache to a virgin state without reallocating the tensors.

        Used by the benchmark to run the continuous and static schedulers back to
        back over the *same* physical block pool (identical starting state).
        """
        for table in self.block_tables:
            for phys in table:
                self.allocator.free(phys)
        self.block_tables[:] = []
        self.seq_lengths[:] = []

    # --------------------------------------------------------------- metrics
    @property
    def num_allocated_blocks(self) -> int:
        return self.allocator.num_allocated

    @property
    def logical_tokens(self) -> int:
        return sum(self.seq_lengths)

    @property
    def block_utilization(self) -> float:
        """Fraction of allocated block *slots* actually holding a token."""
        if self.num_allocated_blocks == 0:
            return 0.0
        return float(self.logical_tokens) / float(
            self.num_allocated_blocks * self.block_size
        )

    @property
    def kv_memory_bytes(self) -> int:
        return 2 * self._k.numel() * self._k.element_size()

    # ------------------------------------------------------ read (for attention)
    def key_map(self, layer: int, phys: List[int]) -> torch.Tensor:
        """Gather the physical blocks of ``layer`` for a list of physical ids."""
        return self._k[layer][phys]  # (num_selected, block_size, kv_heads, head_dim)

    def value_map(self, layer: int, phys: List[int]) -> torch.Tensor:
        return self._v[layer][phys]

    def kv_stats(self) -> Optional[Dict[str, float]]:
        if self.num_sequences == 0:
            return None
        return {
            "num_allocated_blocks": self.num_allocated_blocks,
            "logical_tokens": self.logical_tokens,
            "block_utilization": round(self.block_utilization, 4),
            "kv_memory_bytes": self.kv_memory_bytes,
            "max_seq_len": max(self.seq_lengths),
        }