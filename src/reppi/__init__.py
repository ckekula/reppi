"""
reppi — Representation Learning Algorithms
==========================================

A Python library implementing classical sparse representation and
dictionary learning algorithms.

Modules
-------
sparse
    Sparse coding (OMP, Batch-OMP).
dictionary
    Dictionary learning (K-SVD, LC-KSVD1, LC-KSVD2).
"""

from reppi.dictionary import (
    FDDL,
    KSVD,
    LCKSVD,
    FrozenDictionaryLearner,
    IncrementalFrozenDictionary,
)
from reppi.sparse import FISTA, OMP, fista_core

__all__ = ["FDDL", "FISTA", "KSVD", "LCKSVD", "OMP", "FrozenDictionaryLearner", "IncrementalFrozenDictionary", "fista_core"]
__version__ = "0.1.56"