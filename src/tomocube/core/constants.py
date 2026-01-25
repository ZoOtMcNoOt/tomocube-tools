"""
TCF Constants - Shared constants for Tomocube TCF file handling.

This module centralizes all magic numbers and path strings used throughout
the codebase for easier maintenance and consistency.
"""

from __future__ import annotations

# =============================================================================
# Default Resolutions (µm/pixel)
# =============================================================================
# These are typical values for Tomocube HT-2H system with 60x objective.
# Actual values are read from file metadata when available.

# Holotomography (HT) defaults
DEFAULT_HT_RES_X: float = 0.196  # µm/pixel
DEFAULT_HT_RES_Y: float = 0.196  # µm/pixel
DEFAULT_HT_RES_Z: float = 0.839  # µm/pixel (axial)

# Fluorescence (FL) defaults
DEFAULT_FL_RES_X: float = 0.122  # µm/pixel
DEFAULT_FL_RES_Y: float = 0.122  # µm/pixel
DEFAULT_FL_RES_Z: float = 1.044  # µm/pixel (axial)


# =============================================================================
# HDF5 Path Constants
# =============================================================================
# Standard paths within TCF (HDF5) file structure

# Data paths
PATH_DATA_3D: str = "Data/3D"
PATH_DATA_2D_MIP: str = "Data/2DMIP"
PATH_DATA_3D_FL: str = "Data/3DFL"

# Metadata paths
PATH_INFO_DEVICE: str = "Info/Device"
PATH_METADATA_CONFIG: str = "Info/MetaData/RawData/Config"
PATH_METADATA_EXPERIMENT: str = "Info/MetaData/RawData/Experiment"
PATH_FL_REGISTRATION: str = "Info/MetaData/FL/Registration"

# Attribute names
ATTR_RESOLUTION_X: str = "ResolutionX"
ATTR_RESOLUTION_Y: str = "ResolutionY"
ATTR_RESOLUTION_Z: str = "ResolutionZ"
ATTR_RI_MIN: str = "RIMin"
ATTR_RI_MAX: str = "RIMax"
ATTR_MAGNIFICATION: str = "Magnification"
ATTR_NA: str = "NA"
ATTR_RI: str = "RI"
ATTR_OFFSET_Z: str = "OffsetZ"

# Registration attributes
ATTR_REG_ROTATION: str = "Rotation"
ATTR_REG_SCALE: str = "Scale"
ATTR_REG_TRANSLATION_X: str = "TranslationX"
ATTR_REG_TRANSLATION_Y: str = "TranslationY"


# =============================================================================
# Default Timepoint
# =============================================================================
DEFAULT_TIMEPOINT: str = "000000"


# =============================================================================
# Visualization Defaults
# =============================================================================
DEFAULT_PERCENTILE_LOW: float = 1.0
DEFAULT_PERCENTILE_HIGH: float = 99.0
