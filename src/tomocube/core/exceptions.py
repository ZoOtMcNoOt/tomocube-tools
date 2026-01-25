"""
TCF Exceptions - Custom exception classes for TCF file handling.

Exception Hierarchy:
    TCFError (base)
    ├── TCFFileError - File access or format errors
    ├── TCFParseError - Metadata parsing errors
    └── TCFNoFluorescenceError - Missing fluorescence data
"""

from __future__ import annotations


class TCFError(Exception):
    """Base exception for all TCF-related errors."""


class TCFFileError(TCFError):
    """Exception raised for file access or format errors.
    
    Examples:
        - File not found
        - Invalid HDF5 format
        - Missing required datasets
    """


class TCFParseError(TCFError):
    """Exception raised for metadata parsing errors.
    
    Examples:
        - Invalid JSON in metadata
        - Missing required attributes
        - Unexpected data format
    """


class TCFNoFluorescenceError(TCFError):
    """Exception raised when fluorescence data is required but not present."""
