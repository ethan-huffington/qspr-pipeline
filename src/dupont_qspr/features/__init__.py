"""Featurization - two parallel paths, one shared cache mechanism.

``cache`` provides persistent, content-addressed storage keyed by canonical
SMILES. ``descriptors`` supplies Path A (RDKit 2D descriptors + ECFP4) feeding the
XGBoost track; ``encoders`` supplies Path B (frozen transformer embeddings)
feeding the multi-task neural heads.

Path B caches *two* encoders rather than one. ChemBERTa-2 is the encoder the brief
specifies, but its published tokenizer cannot emit multi-character tokens, so it
reads chlorine as carbon and drops stereochemistry - and it was pretrained that
way, so the defect is not repairable. Caching a second encoder with a working
tokenizer lets step 7 separate "does multi-task learning help?" from "what did the
tokenizer cost?" instead of confounding the two.

Everything is computed once per molecule and reused by every fold and every tuning
trial, which is what makes nested cross-validation affordable here.
"""

from __future__ import annotations

from dupont_qspr.features.cache import CacheStats, DiskFeatureCache
from dupont_qspr.features.descriptors import build_descriptor_cache
from dupont_qspr.features.encoders import build_encoder_cache, build_encoder_caches

__all__ = [
    "CacheStats",
    "DiskFeatureCache",
    "build_descriptor_cache",
    "build_encoder_cache",
    "build_encoder_caches",
]
