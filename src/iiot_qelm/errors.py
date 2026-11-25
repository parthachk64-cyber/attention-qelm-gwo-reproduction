"""Exceptions. Messages name the exact missing input and how to fix it."""
from __future__ import annotations


class IIoTError(Exception):
    pass


class SchemaError(IIoTError):
    """Raw file or feature schema does not match the contract."""


class LeakageError(IIoTError):
    """A leakage control would be violated."""


class MissingInputError(IIoTError):
    """A full-scale command needs data or a dependency that is absent."""

    def __init__(self, what: str, how_to_fix: str):
        super().__init__(f"missing: {what}. To fix: {how_to_fix}")
        self.what, self.how_to_fix = what, how_to_fix


class ProvenanceError(IIoTError):
    """A fixture or reduced-budget result was used where a measured one is required."""
