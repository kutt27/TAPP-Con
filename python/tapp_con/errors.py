"""Exception types mirroring the C++ throws in the library, plus the mapping
onto the C-API status codes."""

from __future__ import annotations

from ._types import Status


class TappConError(Exception):
    """Base class for all libtapp_con errors."""


class InvalidParameterError(TappConError):
    """Mirrors std::invalid_argument (bad arguments / constraint parse failure)."""


class OutOfRangeError(TappConError):
    """Mirrors std::out_of_range (block extends beyond tensor bounds)."""


class BlockOverlapError(TappConError):
    """Raised when a structural invariant is violated."""


class BackendError(TappConError):
    """Mirrors the catch-all backend failure."""


_STATUS_BY_EXCEPTION = (
    (MemoryError, Status.MEMORY_ERROR),
    (InvalidParameterError, Status.INVALID_PARAMETER),
    (ValueError, Status.INVALID_PARAMETER),
    (IndexError, Status.INVALID_PARAMETER),
    (OutOfRangeError, Status.INVALID_PARAMETER),
    (BlockOverlapError, Status.BLOCK_OVERLAP),
)


def status_for(exc: BaseException) -> Status:
    """Map a Python exception to the closest C-API status code (map_exception)."""
    for exc_type, status in _STATUS_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return status
    return Status.BACKEND_ERROR
