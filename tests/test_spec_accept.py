"""Tests for the speculation accept/reject helper and the KV rollback primitive.

Two independent concerns are checked:

* :func:`batch_accept` -- on greedy (one-hot) distributions every coin flip is exact,
  so both the full-acceptance (bonus) path and the rejection (residual resample)
  path are deterministic and directly hand-verifiable.
* ``PagedKVCache.truncate`` -- rewinding a cache returns the freed physical blocks
  to the allocator and shrinks ``seq_lengths``, which is the engine's rollback hook.
"""

from __future__ import annotations

import torch
import pytest

from engine.kv import PagedKVCache
from engine.spec_scheduler import batch_accept

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _one_hot(token: int, vocab: int = 8) -> torch.Tensor:
    vec = torch.zeros(vocab, device=_DEVICE)
    vec[token] = 1.0
    return vec


def test_batch_accept_full_acceptance_with_bonus() -> None:
    generator = torch.Generator().manual_seed(0)
    token = 3
    gamma = 2
    # Target's greedy argmax matches the draft at every proposal position, plus a
    # bonus draw from p's bonus row (also greedy argmax).
    p_by = {
        0: torch.stack([_one_hot(token), _one_hot(token), _one_hot(token)])
    }  # (gamma+1, V)
    q_by = {0: [_one_hot(token), _one_hot(token)]}
    proposals = {0: [token, token]}

    accepted = batch_accept(p_by, q_by, proposals, generator)
    assert accepted[0] == [token, token, token]  # gamma proposals + 1 bonus


def test_batch_accept_rejection_resamples_residual() -> None:
    generator = torch.Generator().manual_seed(0)
    target = 5   # target's argmax
    draft = 2    # draft proposes a DIFFERENT token
    # p one-hot on `target` => acceptance of `draft` is 0 -> reject, resample (p-q)_+ = p.
    p_by = {0: torch.stack([_one_hot(target), _one_hot(target)])}  # (2, V) gamma+1
    q_by = {0: [_one_hot(draft)]}
    proposals = {0: [draft]}

    accepted = batch_accept(p_by, q_by, proposals, generator)
    assert accepted[0] == [target]  # the corrected token, and no bonus will be drawn


def test_batch_accept_rejects_then_stops() -> None:
    generator = torch.Generator().manual_seed(0)
    target = 4
    good = 1
    bad = 7
    # First proposal matches (accept), second mismatches (reject -> resample target).
    p_by = {
        0: torch.stack(
            [_one_hot(good), _one_hot(target), _one_hot(target)]
        )
    }
    q_by = {0: [_one_hot(good), _one_hot(bad)]}
    proposals = {0: [good, bad]}

    accepted = batch_accept(p_by, q_by, proposals, generator)
    assert accepted[0] == [good, target]  # stop right after the rejection


def test_kv_truncate_releases_physical_blocks() -> None:
    kv = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        capacity_blocks=32,
        block_size=4,
        device=_DEVICE,
        dtype=torch.float32,
    )
    kv.add_sequence(0)
    num_written = 10
    for pos in range(num_written):
        kv.write(0, 0, pos, torch.randn(2, 8, device=_DEVICE), torch.randn(2, 8, device=_DEVICE))
    assert kv.seq_lengths[0] == 10
    blocks_full = kv.num_allocated_blocks

    kv.truncate(0, 4)
    assert kv.seq_lengths[0] == 4
    blocks_rewound = kv.num_allocated_blocks
    # 10 tokens -> ceil(10/4)=3 blocks; 4 tokens -> ceil(4/4)=1 block; 2 freed.
    assert blocks_full - blocks_rewound == 2
    assert blocks_rewound == 1