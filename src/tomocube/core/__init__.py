"""
Tomocube core module - File I/O, types, and exceptions.

This module provides the foundational classes for working with TCF files:
    - TCFFile: Metadata container for TCF file information
    - TCFFileLoader: File I/O and data loading
    - RegistrationParams: FL-to-HT registration parameters
    - ViewerState: Mutable viewer state container
    - Custom exceptions for error handling
"""

from tomocube.core.constants import (
    # Resolutions
    DEFAULT_HT_RES_X,
    DEFAULT_HT_RES_Y,
    DEFAULT_HT_RES_Z,
    DEFAULT_FL_RES_X,
    DEFAULT_FL_RES_Y,
    DEFAULT_FL_RES_Z,
    # Paths
    PATH_DATA_3D,
    PATH_DATA_2D_MIP,
    PATH_DATA_3D_FL,
    PATH_INFO_DEVICE,
    PATH_METADATA_CONFIG,
    PATH_METADATA_EXPERIMENT,
    PATH_FL_REGISTRATION,
    # Attributes
    ATTR_RESOLUTION_X,
    ATTR_RESOLUTION_Y,
    ATTR_RESOLUTION_Z,
    ATTR_RI_MIN,
    ATTR_RI_MAX,
    ATTR_MAGNIFICATION,
    ATTR_NA,
    ATTR_RI,
    ATTR_OFFSET_Z,
    ATTR_REG_ROTATION,
    ATTR_REG_SCALE,
    ATTR_REG_TRANSLATION_X,
    ATTR_REG_TRANSLATION_Y,
    # Defaults
    DEFAULT_TIMEPOINT,
    DEFAULT_PERCENTILE_LOW,
    DEFAULT_PERCENTILE_HIGH,
)
from tomocube.core.exceptions import (
    TCFError,
    TCFFileError,
    TCFParseError,
    TCFNoFluorescenceError,
)
from tomocube.core.types import (
    RegistrationParams,
    ViewerState,
)
from tomocube.core.file import (
    TCFFile,
    TCFFileLoader,
)

__all__ = [
    # Constants - Resolutions
    "DEFAULT_HT_RES_X",
    "DEFAULT_HT_RES_Y",
    "DEFAULT_HT_RES_Z",
    "DEFAULT_FL_RES_X",
    "DEFAULT_FL_RES_Y",
    "DEFAULT_FL_RES_Z",
    # Constants - Paths
    "PATH_DATA_3D",
    "PATH_DATA_2D_MIP",
    "PATH_DATA_3D_FL",
    "PATH_INFO_DEVICE",
    "PATH_METADATA_CONFIG",
    "PATH_METADATA_EXPERIMENT",
    "PATH_FL_REGISTRATION",
    # Constants - Attributes
    "ATTR_RESOLUTION_X",
    "ATTR_RESOLUTION_Y",
    "ATTR_RESOLUTION_Z",
    "ATTR_RI_MIN",
    "ATTR_RI_MAX",
    "ATTR_MAGNIFICATION",
    "ATTR_NA",
    "ATTR_RI",
    "ATTR_OFFSET_Z",
    "ATTR_REG_ROTATION",
    "ATTR_REG_SCALE",
    "ATTR_REG_TRANSLATION_X",
    "ATTR_REG_TRANSLATION_Y",
    # Constants - Defaults
    "DEFAULT_TIMEPOINT",
    "DEFAULT_PERCENTILE_LOW",
    "DEFAULT_PERCENTILE_HIGH",
    # Exceptions
    "TCFError",
    "TCFFileError",
    "TCFParseError",
    "TCFNoFluorescenceError",
    # Types
    "RegistrationParams",
    "ViewerState",
    # File
    "TCFFile",
    "TCFFileLoader",
]
