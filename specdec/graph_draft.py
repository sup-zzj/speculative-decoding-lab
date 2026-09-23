"""CUDA-graph accelerated draft engine for speculative decoding.

The draft model is called one token at a time while its KV cache grows.  The
default ``DynamicCache`` allocates fresh tensors every step and every forward
launches the same kernel sequence with per-launch overhead, which is the bulk
of the 47 ms/token measured on the 0.5B draft.  A CUDA Graph can only capture
*fixed* tensor shapes, so this engine re-hosts the draft on a transformers
``StaticCache`` (fixed-size key/value buffers) and captures a single
"advance one token" graph.  Each replay submits the whole forward as one unit,
eliminating the per-kernel launch overhead.

Rewinding -- the rejection path of speculative decoding -- is implemented by
moving a logical position pointer backwards.  The causal mask is derived from
``cache_position`` inside the model, so stale entries beyond the pointer are
masked out even though the underlying buffer still holds them.  No physical
truncation is needed, which is what makes the fixed-shape cache legal.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from transformers import StaticCache

from .config import SamplingConfig
from .models import CausalLM
from .sampler import logits_to_probs, sample_token
from .utils import setup_logger

logger = setup_logger(__name__)


class GraphDraft:
    """A draft model whose single-token advance runs as one CUDA Graph."""

    def __init__(self, lm: CausalLM, max_cache_len: int) -> None:
        if lm.device.type != "cuda":
            raise ValueError("GraphDraft requires a CUDA-placed model")
        self.lm = lm
        self.device = lm.device
        self.max_cache_len = int(max_cache_len)
        self.forward_calls = 0

        self._cache = StaticCache(
            lm.model.config,
            batch_size=1,
            max_cache_len=self.max_cache_len,
            device=self.device,
            dtype=lm.dtype,
        )
        self._position = 0  # logical filled length, not necessarily a valid KV row

        # Static buffers for the captured graph.  Their *contents* change
        # between replays but their shapes never do, which is the contract a
        # CUDA Graph requires.
        self._input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._cache_position = torch.zeros(1, dtype=torch.long, device=self.device)
        self._attention_mask = torch.ones(
            (1, self.max_cache_len), dtype=torch.long, device=self.device
        )
        self._logits = torch.zeros(
            (1, 1, lm.vocab_size), dtype=lm.dtype, device=self.device
        )

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    # ------------------------------------------------------------------ graph
    def _capture(self) -> None:
        """Run the single-token forward on a warm stream and capture it.

        The capture itself is a real execution, so it must not touch any
        position that real data will occupy.  We therefore run the warm-up and
        the capture against the *tail* slot (``max_cache_len - 1``) and zero the
        whole cache afterwards; ``prefill`` repopulates it from a clean state.
        """

        def run() -> None:
            out = self.lm.model(
                input_ids=self._input_ids,
                attention_mask=self._attention_mask,
                past_key_values=self._cache,
                use_cache=True,
                cache_position=self._cache_position,
            )
            self._logits.copy_(out.logits)

        # Point the graph at the scratch slot and the zero token for the
        # warm-up/capture executions.  ``_step`` overwrites both before every
        # replay, so the captured values themselves are irrelevant.
        self._input_ids[0, 0] = 0
        self._cache_position[0] = self.max_cache_len - 1

        # Warm up the allocator on a side stream so the capture itself has no
        # allocation surprises, then capture on the default stream.
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warm)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            run()

        # Drop the scratch write so the graph starts from a clean cache.
        self._cache.reset()
        self._position = 0

    def _ensure_captured(self) -> None:
        if self._graph is None:
            self._capture()
            logger.info(
                "draft graph captured (max_cache_len=%d, vocab=%d)",
                self.max_cache_len,
                self.lm.vocab_size,
            )

    # -------------------------------------------------------------- interface
    def prefill(self, prompt_ids: Sequence[int]) -> None:
        """Feed the prompt with an eager forward, then rewind one position.

        The last prompt token is left *un-committed* as the speculative
        decoder's "pending" token, matching the invariant used elsewhere.
        """
        self._ensure_captured()
        self._cache.reset()
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        n = ids.shape[1]
        cpos = torch.arange(n, dtype=torch.long, device=self.device)
        mask = torch.ones((1, n), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.lm.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=self._cache,
                use_cache=True,
                cache_position=cpos,
            )
        self._position = n
        self.forward_calls += 1

    def rewind(self, length: int) -> None:
        """Logical truncation: the next token is written at ``length``."""
        self._position = max(0, int(length))

    def cache_len(self) -> int:
        return self._position

    def propose(
        self,
        pending_token: int,
        gamma: int,
        sampling: SamplingConfig,
        generator: Optional[torch.Generator],
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """Advance ``gamma`` tokens, returning proposals and draft probs."""
        self._ensure_captured()
        proposals: List[int] = []
        rows: List[torch.Tensor] = []
        token = pending_token
        for _ in range(gamma):
            logits = self._step(token)
            q = logits_to_probs(logits, sampling)
            rows.append(q)
            token = sample_token(q, generator)
            proposals.append(token)
        return proposals, rows

    def step_token(self, token_id: int) -> None:
        """Feed a single committed token without returning logits (cache sync)."""
        self._ensure_captured()
        self._step(token_id)

    # ------------------------------------------------------------------ core
    def _step(self, token_id: int) -> torch.Tensor:
        self._input_ids[0, 0] = token_id
        self._cache_position[0] = self._position
        self._graph.replay()  # type: ignore[union-attr]
        self._position += 1
        self.forward_calls += 1
        # Copy off the static buffer: the caller may keep the tensor alive
        # after the next replay overwrites it.
        return self._logits[0, 0].clone()
