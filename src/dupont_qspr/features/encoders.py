"""Path B - frozen ChemBERTa-2 embeddings, the representation the neural track sees.

The encoder is **frozen**, and that is a design decision rather than a default
left unrevisited. With a few thousand labels per property, fine-tuning a
transformer overfits comfortably before it learns anything transferable. Freezing
also has a second effect that shapes the whole project: because the trunk never
changes, an embedding depends only on the molecule, so it can be computed once and
cached. Every fold and every Optuna trial then trains only small heads on stored
vectors, which is what makes nested cross-validation cheap enough to run at all.

Fine-tuning would break both halves of that. The embeddings would become a
function of the current fold's training data, the cache would be invalid, and the
representation would leak across the fold boundary.

**Checkpoint choice.** ``DeepChem/ChemBERTa-77M-MLM`` is the default rather than
its ``-MTR`` sibling. MTR was pretrained by regressing computed RDKit descriptors,
several of which are logP estimates - so using it would let the lipophilicity task
benefit from pretraining targets that are near-duplicates of its own label, and
the comparison against the descriptor track would stop being fair.

**Pooling.** Mean over non-padding tokens by default. RoBERTa-style models have no
next-sentence objective, so their ``<s>`` token is not trained to summarise a
sequence the way BERT's ``[CLS]`` is; mean pooling is the better-behaved choice
here. Both are available, and whichever is used is recorded with the cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from dupont_qspr.config import Config
from dupont_qspr.features.cache import DiskFeatureCache

__all__ = [
    "build_encoder_cache",
    "build_encoder_caches",
    "compute_path_b",
    "embedding_width",
    "resolve_device",
]


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device. ``auto`` prefers MPS on Apple silicon, then CUDA."""
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@cache
def _load_encoder(model_name: str, device: str) -> tuple[Any, Any]:
    """Load tokenizer and encoder once per process, in eval mode.

    ``AutoModel`` rather than ``AutoModelForMaskedLM``: we want the encoder trunk,
    not the language-modelling head, which is discarded.

    ``add_pooling_layer=False`` matters more than it looks. Loading this checkpoint
    without it makes transformers attach a pooler whose weights are *not* in the
    checkpoint and are therefore randomly initialised - and then report that fact
    in a wall of text nobody reads. We pool manually from ``last_hidden_state``, so
    it was never used, but leaving a randomly initialised layer attached to a model
    described as "frozen pretrained" is an accident waiting to be made.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    # The load report is long, printed to stderr, and entirely expected here:
    # the masked-LM head is discarded on purpose. Silencing it keeps a real
    # warning visible if one ever appears.
    transformers.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    except TypeError:
        # Not every architecture accepts the flag; fall back and rely on the fact
        # that we only ever read last_hidden_state.
        model = AutoModel.from_pretrained(model_name)

    model.eval()
    # eval() disables dropout; requires_grad_(False) makes it impossible to
    # accidentally accumulate gradients into the trunk later. "Frozen" should be
    # enforced by the object, not by remembering to avoid it.
    model.requires_grad_(False)
    model.to(torch.device(device))
    return tokenizer, model


def embedding_width(model_name: str) -> int:
    """Hidden size of the checkpoint, read from its config rather than assumed."""
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(model_name).hidden_size)


def compute_path_b(
    smiles: Sequence[str],
    *,
    model_name: str,
    device: str = "auto",
    batch_size: int = 64,
    pooling: str = "mean",
    max_length: int = 512,
) -> npt.NDArray[np.float32]:
    """Embed a batch of SMILES with the frozen encoder."""
    import torch

    if not smiles:
        return np.empty((0, embedding_width(model_name)), dtype=np.float32)

    resolved = resolve_device(device)
    tokenizer, model = _load_encoder(model_name, resolved)
    torch_device = torch.device(resolved)

    outputs: list[npt.NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = list(smiles[start : start + batch_size])
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(torch_device)

            hidden = model(**encoded).last_hidden_state  # (b, tokens, width)

            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                # Mean over real tokens only. Averaging over padding would make a
                # molecule's embedding depend on the longest SMILES that happened
                # to share its batch, which would silently break cache identity.
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            outputs.append(pooled.to("cpu", torch.float32).numpy())

    return np.vstack(outputs)


def build_encoder_cache(cfg: Config, key: str | None = None) -> DiskFeatureCache:
    """Wire one frozen encoder into a persistent cache.

    ``key`` names an entry in ``features.encoders``; omitting it uses
    ``features.primary_encoder``. The key becomes part of the cache filename, so
    two encoders coexist on disk without any chance of their vectors mixing.
    """
    key = key or cfg.features.primary_encoder
    if key not in cfg.features.encoders:
        raise KeyError(
            f"unknown encoder {key!r}; configured encoders are "
            f"{sorted(cfg.features.encoders)}"
        )

    model_name = cfg.features.encoders[key]
    device = resolve_device(cfg.features.device)
    width = embedding_width(model_name)

    return DiskFeatureCache(
        name=f"path_b_{key}",
        version=cfg.features.version,
        n_features=width,
        compute=lambda batch: compute_path_b(
            batch,
            model_name=model_name,
            device=device,
            batch_size=cfg.features.chemberta_batch_size,
            pooling=cfg.features.chemberta_pooling,
        ),
        directory=cfg.features_dir,
        feature_names=tuple(f"{key}_{i}" for i in range(width)),
        recipe={
            "model": model_name,
            "pooling": cfg.features.chemberta_pooling,
            "frozen": True,
            "device": device,
            "hidden_size": width,
            "is_primary": key == cfg.features.primary_encoder,
        },
    )


def build_encoder_caches(cfg: Config) -> dict[str, DiskFeatureCache]:
    """One cache per configured encoder, so step 7 can compare representations."""
    return {key: build_encoder_cache(cfg, key) for key in cfg.features.encoders}
