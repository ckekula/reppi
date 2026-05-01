"""Dictionary learning algorithms."""

from reppi.dictionary.ksvd import KSVD
from reppi.dictionary.lc_ksvd import LCKSVD, initialization4lcksvd

__all__ = ["KSVD", "LCKSVD", "initialization4lcksvd"]