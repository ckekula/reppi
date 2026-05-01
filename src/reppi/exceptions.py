"""
Custom exceptions for the reppi library.
"""


class ReppiError(Exception):
    """Base exception for all reppi errors."""


class DictionaryNormalizationError(ReppiError):
    """Raised when dictionary atoms are not unit-norm."""


class SparseCodingError(ReppiError):
    """Raised when sparse coding fails or receives invalid inputs."""


class DictionaryLearningError(ReppiError):
    """Raised when dictionary learning encounters an unrecoverable error."""


class InvalidParameterError(ReppiError):
    """Raised when an invalid parameter value is supplied."""