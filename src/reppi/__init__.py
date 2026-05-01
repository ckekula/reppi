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
from reppi.dictionary import KSVD, LCKSVD

__all__ = ["OMP", "KSVD", "LCKSVD"]
__version__ = "0.1.0"