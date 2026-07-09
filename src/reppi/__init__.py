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

from reppi.sparse import OMP
from reppi.dictionary import KSVD, LCKSVD, FrozenDictionaryLearner, IncrementalFrozenDictionary

__all__ = ["OMP", "KSVD", "LCKSVD", "FrozenDictionaryLearner", "IncrementalFrozenDictionary"]
__version__ = "0.1.20"