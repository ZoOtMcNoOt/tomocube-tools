"""
TCF Processing - Data processing and transformation utilities.

This module provides functions for:
    - FL-to-HT registration
    - Image normalization
    - Metadata extraction
    - Export to various formats (TIFF, MAT, GIF)
"""

from tomocube.processing.export import (
    export_overlay_gif,
    export_to_gif,
    export_to_mat,
    export_to_png_sequence,
    export_to_tiff,
)
from tomocube.processing.image import (
    compute_overlap_score,
    normalize_image,
    normalize_with_bounds,
)
from tomocube.processing.metadata import (
    discover_related_metadata,
    extract_metadata,
    load_experiment_file,
    load_profile_file,
    load_project_file,
    load_vessel_file,
    parse_ini_string,
)
from tomocube.processing.registration import register_fl_to_ht

__all__ = [
    # Registration
    "register_fl_to_ht",
    # Image processing
    "normalize_image",
    "normalize_with_bounds",
    "compute_overlap_score",
    # Metadata
    "extract_metadata",
    "parse_ini_string",
    "load_profile_file",
    "load_vessel_file",
    "load_project_file",
    "load_experiment_file",
    "discover_related_metadata",
    # Export
    "export_to_tiff",
    "export_to_mat",
    "export_to_png_sequence",
    "export_to_gif",
    "export_overlay_gif",
]
