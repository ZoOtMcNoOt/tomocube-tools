"""
Tomocube Tools - Python library for working with Tomocube TCF files.

This package provides tools for reading, viewing, and processing
holotomography data from Tomocube microscopes.

Modules:
    core: File I/O, types, constants, and exceptions
    processing: FL registration, image normalization, export
    viewer: Interactive visualization tools
"""

from tomocube.core import (
    # File I/O
    TCFFile,
    TCFFileLoader,
    # Types
    RegistrationParams,
    ViewerState,
    # Exceptions
    TCFError,
    TCFFileError,
    TCFParseError,
    TCFNoFluorescenceError,
)
from tomocube.processing import (
    # Registration
    register_fl_to_ht,
    # Image processing
    normalize_image,
    normalize_with_bounds,
    compute_overlap_score,
    # Metadata
    extract_metadata,
    parse_ini_string,
    load_profile_file,
    load_vessel_file,
    load_project_file,
    load_experiment_file,
    discover_related_metadata,
    # Export
    export_to_tiff,
    export_to_mat,
    export_to_png_sequence,
    export_to_gif,
    export_overlay_gif,
)
from tomocube.viewer import (
    TCFViewer,
    SliceViewer,
    FluorescenceMapper,
    MeasurementTool,
    DistanceMeasurement,
    AreaMeasurement,
    extract_line_profile,
)

__version__ = "0.1.0"

__all__ = [
    # Core - File I/O
    "TCFFile",
    "TCFFileLoader",
    # Core - Types
    "RegistrationParams",
    "ViewerState",
    # Core - Exceptions
    "TCFError",
    "TCFFileError",
    "TCFParseError",
    "TCFNoFluorescenceError",
    # Processing
    "register_fl_to_ht",
    "normalize_image",
    "normalize_with_bounds",
    "compute_overlap_score",
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
    # Viewer
    "TCFViewer",
    "SliceViewer",
    "FluorescenceMapper",
    # Measurements
    "MeasurementTool",
    "DistanceMeasurement",
    "AreaMeasurement",
    "extract_line_profile",
]
