"""Dictionary learning algorithms."""

from reppi.dictionary.ksvd import KSVD
from reppi.dictionary.lc_ksvd import LCKSVD, initialization4lcksvd
from reppi.dictionary.frozen import FrozenDictionaryLearner, IncrementalFrozenDictionary
from reppi.dictionary.fddl import FDDL

__all__ = [
    "KSVD",
    "LCKSVD",
    "initialization4lcksvd",
    "FrozenDictionaryLearner",
    "IncrementalFrozenDictionary",
    "FDDL"
]