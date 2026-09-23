"""Thin, cache-aware wrappers around ``transformers`` causal LMs.

The speculative decoder needs to *rewind* the key/value cache (a rejected draft
token must be erased), which the public ``generate`` API does not expose. These
wrappers therefore drive ``model(...)`` directly and keep an explicit
``(cache length, logits)`` contract:

    ``extend(tokens, past)`` feeds ``tokens`` and returns the logits of *every*
    fed position, so the caller can verify a whole draft block in one forward.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import ModelConfig
from .utils import setup_logger

logger = setup_logger(__name__)

PastType = Any  # legacy tuple-of-tuples or transformers Cache object


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Half precision only helps on GPU; CPU inference stays in float32."""
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError("unsupported dtype: {}".format(requested))
    return mapping[requested]


def to_legacy_past(past: PastType) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    """Normalise a cache object into the legacy tuple format we can slice."""
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


def normalize_cache(past: PastType) -> Any:
    """Return a ``DynamicCache``-like object regardless of the input format.

    ``transformers`` accepts the legacy tuple-of-tuples format but deprecates it,
    and more importantly the legacy format gives us no ``crop`` primitive, which
    is the operation the whole speculative decoder is built around.
    """
    if past is None:
        return None
    if hasattr(past, "crop"):
        return past
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(past)


def truncate_past(past: PastType, length: int) -> Optional[Any]:
    """Physically drop every cache entry beyond ``length`` positions.

    This is the operation that makes speculative decoding correct: after a draft
    token is rejected, the target's cache must forget the rejected token and
    everything that followed it.
    """
    cache = normalize_cache(past)
    if cache is None or length <= 0:
        return None
    if past_length(cache) <= length:
        return cache
    cropped = cache.crop(length)
    # 4.46 crops in place and returns None; <= 4.42 returned a tuple of caches.
    if isinstance(cropped, (tuple, list)):
        cropped = cropped[0]
    return cropped if cropped is not None else cache


def past_length(past: PastType) -> int:
    """Number of positions currently held in the cache."""
    cache = normalize_cache(past)
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    legacy = to_legacy_past(cache)
    return int(legacy[0][0].shape[-2])


class CausalLM:
    """A single causal language model with manual cache control."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
        name: str = "model",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.name = name
        self.forward_calls = 0  # instrumented: the benchmark counts real forwards

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pretrained(cls, config: ModelConfig, name: str = "model") -> "CausalLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        cache_dir = os.path.abspath(config.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        model_id = config.target_model if name == "target" else config.draft_model

        # A fully local checkpoint (e.g. downloaded with curl) takes precedence
        # over the HuggingFace hub layout; the model is otherwise resolved
        # through the hub cache under ``cache_dir``.
        local_path = cls._resolve_local_checkpoint(cache_dir, model_id)
        source = local_path or model_id
        if local_path:
            logger.info("loading %s from local checkpoint %s", name, source)

        logger.info("loading %s (%s) -> %s / %s", name, model_id, device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return cls(model, tokenizer, device, dtype, name=name)

    @classmethod
    def _resolve_local_checkpoint(cls, cache_dir: str, model_id: str) -> Optional[str]:
        """Return a local checkpoint directory for ``model_id`` if one exists.

        Supports both ``models/Qwen/Qwen2.5-0.5B`` and the flat
        ``models/Qwen2.5-0.5B`` layout; a candidate is only accepted once its
        weights are on disk. Models larger than the safetensors 2 GB single-file
        limit are shipped as shards (``model-00001-of-00002.safetensors``), so a
        candidate is valid when either the single ``model.safetensors`` file or
        at least one weight shard is present.
        """
        candidates = [
            os.path.join(cache_dir, model_id),
            os.path.join(cache_dir, os.path.basename(model_id)),
        ]
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            if os.path.exists(os.path.join(candidate, "model.safetensors")):
                return candidate
            if glob.glob(os.path.join(candidate, "model-*-of-*.safetensors")):
                return candidate
        return None

    @classmethod
    def from_module(
        cls, model: torch.nn.Module, tokenizer: Any, device: torch.device, dtype: torch.dtype, name: str
    ) -> "CausalLM":
        return cls(model.to(device).to(dtype), tokenizer, device, dtype, name=name)

    # -------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters_M": round(self.num_params / 1e6, 1),
            "layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
        }

    # ------------------------------------------------------------------ codec
    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def current_cache_length(self, past: PastType) -> int:
        return past_length(past)

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def prefill(self, token_ids: Sequence[int]) -> Tuple[PastType, torch.Tensor]:
        """Feed a prompt and return ``(cache, logits_of_last_position)``."""
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self.forward_calls += 1
        # ``prefill`` always returns a modern cache object: everything downstream
        # relies on ``crop`` being available.
        return normalize_cache(out.past_key_values), out.logits[:, -1, :]

    @torch.no_grad()
    def extend(
        self, token_ids: Sequence[int], past: PastType
    ) -> Tuple[PastType, torch.Tensor]:
        """Feed one *block* of tokens; return the cache and per-position logits.

        The returned tensor has shape ``(len(token_ids), vocab)``: row ``i`` is
        the distribution over the token that follows the ``i``-th fed token.
        """
        if not token_ids:
            raise ValueError("extend() requires at least one token")
        past = normalize_cache(past)
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        prefix = past_length(past)
        attention_mask = torch.ones(
            (1, prefix + input_ids.shape[1]), dtype=torch.long, device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        self.forward_calls += 1
        return normalize_cache(out.past_key_values), out.logits[0]

    @torch.no_grad()
    def prefill_batch(
        self, batch: Sequence[Sequence[int]]
    ) -> Tuple[PastType, torch.Tensor]:
        """Prefill ``batch`` (``B`` sequences) and return their batched KV cache.

        Inputs may differ in length; rows are *left*-padded with
        ``pad_token_id`` so the ``[-1]`` position is the last real token of every
        row. The returned logits have shape ``(B, vocab)``: row ``i`` is the
        distribution over the token following the ``i``-th sequence. Does not
        accept an existing cache (used as the first forward of a batch).
        """
        if not batch:
            raise ValueError("prefill_batch() requires at least one sequence")
        lengths = [len(seq) for seq in batch]
        max_len = max(lengths)
        if max_len == 0:
            raise ValueError("prefill_batch() sequences must be non-empty")
        padded = torch.full(
            (len(batch), max_len),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        for index, seq in enumerate(batch):
            offset = max_len - len(seq)
            padded[index, offset:] = torch.as_tensor(seq, dtype=torch.long, device=self.device)
        # Left padding must be masked out of attention, so real tokens stay causal.
        attention_mask = torch.zeros_like(padded)
        for index, length in enumerate(lengths):
            attention_mask[index, max_len - length :] = 1
        out = self.model(
            input_ids=padded,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self.forward_calls += 1
        return normalize_cache(out.past_key_values), out.logits[:, -1, :]

    @torch.no_grad()
    def extend_batch(
        self, token_ids_2d: torch.Tensor, past: PastType
    ) -> Tuple[PastType, torch.Tensor]:
        """Extend ``B`` cached sequences by a fixed block; return cache and logits.

        ``token_ids_2d`` has shape ``(B, L)`` and holds ``L`` *new* tokens per
        row (no padding inside the block). Returns the new cache plus per-position
        logits of shape ``(B, L, vocab)``: ``[b, i]`` is the distribution over the
        token that follows the ``i``-th newly fed token of row ``b``. All rows in
        the cache must already share one batch dimension, which is the invariant
        :meth:`prefill_batch` establishes.
        """
        past = normalize_cache(past)
        input_ids = token_ids_2d.to(device=self.device, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, block_len = input_ids.shape
        prefix = past_length(past)
        attention_mask = torch.ones(
            (batch_size, prefix + block_len), dtype=torch.long, device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        self.forward_calls += 1
        return normalize_cache(out.past_key_values), out.logits

    @torch.no_grad()
    def logits_for_next(self, token_id: int, past: PastType) -> Tuple[PastType, torch.Tensor]:
        """Advance the cache by exactly one token and return the next logits."""
        new_past, logits = self.extend([token_id], past)
        return new_past, logits[-1]

    def reset_counters(self) -> None:
        self.forward_calls = 0


# --------------------------------------------------------------------- toy LM
class ToyTokenizer:
    """Character-level tokenizer covering ``vocab_size`` distinct symbols."""

    def __init__(self, vocab_size: int = 8) -> None:
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


def build_toy_pair(
    vocab_size: int = 8,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    noise: float = 0.05,
    logit_scale: float = 6.0,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[CausalLM, CausalLM, ToyTokenizer]:
    """Two randomly initialised tiny transformers that share a vocabulary.

    The draft is initialised as the target plus Gaussian noise of scale
    ``noise``. That gives a *controlled* acceptance rate: as ``noise`` grows the
    two distributions drift apart, which lets us validate the analytic speed-up
    model without downloading a single real checkpoint.

    ``logit_scale`` sharpens the randomly initialised output head. Untouched
    random weights produce a nearly uniform next-token distribution, and a
    near-uniform target makes every distribution test statistically meaningless
    (the Monte-Carlo noise floor swamps the signal).
    """
    from transformers import AutoModelForCausalLM, Qwen2Config

    device = device or resolve_device("auto")
    dtype = torch.float32  # tiny models are faster in fp32 everywhere

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=max(1, num_heads // 2),
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )

    torch.manual_seed(seed)
    target_module = AutoModelForCausalLM.from_config(config).eval()
    if logit_scale != 1.0:
        with torch.no_grad():
            target_module.lm_head.weight.mul_(logit_scale)

    torch.manual_seed(seed + 1)
    draft_module = AutoModelForCausalLM.from_config(config).eval()
    draft_module.load_state_dict(target_module.state_dict())

    if noise > 0:
        generator = torch.Generator().manual_seed(seed + 2)
        with torch.no_grad():
            for parameter in draft_module.parameters():
                jitter = torch.randn(parameter.shape, generator=generator) * noise
                parameter.add_(jitter)

    tokenizer = ToyTokenizer(vocab_size=vocab_size)
    target = CausalLM.from_module(target_module, tokenizer, device, dtype, name="target")
    draft = CausalLM.from_module(draft_module, tokenizer, device, dtype, name="draft")
    return target, draft, tokenizer