"""
TCF Metadata - Metadata extraction and parsing utilities.

This module provides functions for extracting and parsing
metadata from TCF files and external Tomocube metadata files.

Supported external formats:
    - .prf (INI format) - Processing profiles
    - .vessel (JSON) - Well plate geometry
    - .tcxpro (JSON) - Project metadata
    - .tcxexp (JSON) - Experiment metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tomocube.core.constants import (
    PATH_INFO_DEVICE,
    PATH_METADATA_CONFIG,
    PATH_METADATA_EXPERIMENT,
)
from tomocube.core.file import _as_dataset, _as_group

logger = logging.getLogger(__name__)


def extract_metadata(f: h5py.File) -> dict[str, Any]:
    """
    Extract all metadata from a TCF file.

    Args:
        f: Open HDF5 file handle

    Returns:
        Dictionary containing all extracted metadata
    """
    metadata: dict[str, Any] = {}

    # Config (INI format)
    if PATH_METADATA_CONFIG in f:
        try:
            config_ds = _as_dataset(f[PATH_METADATA_CONFIG])
            config_bytes = np.asarray(config_ds)
            config_str = bytes(config_bytes).decode("utf-8", errors="ignore")
            metadata["config"] = parse_ini_string(config_str)
        except Exception:
            logger.debug("Failed to parse config metadata", exc_info=True)

    # Experiment (JSON format)
    if PATH_METADATA_EXPERIMENT in f:
        try:
            exp_ds = _as_dataset(f[PATH_METADATA_EXPERIMENT])
            exp_bytes = np.asarray(exp_ds)
            exp_str = bytes(exp_bytes).decode("utf-8", errors="ignore")
            metadata["experiment"] = json.loads(exp_str)
        except Exception:
            logger.debug("Failed to parse experiment metadata", exc_info=True)

    # Device parameters
    if PATH_INFO_DEVICE in f:
        dev = _as_group(f[PATH_INFO_DEVICE])
        device: dict[str, Any] = {}
        for key in dev.attrs.keys():
            val = np.asarray(dev.attrs[key])
            if hasattr(val, "__len__") and len(val) == 1:
                device[key] = val[0]
            else:
                device[key] = val
        metadata["device"] = device

    return metadata


def parse_ini_string(ini_str: str) -> dict[str, Any]:
    """
    Parse INI-format string into nested dictionary.

    Args:
        ini_str: INI-formatted configuration string

    Returns:
        Dictionary with section names as keys, containing key-value pairs.
        Values are automatically converted to int/float when possible.
    """
    result: dict[str, Any] = {}
    current_section = "_global"

    for line in ini_str.split("\n"):
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            result[current_section] = {}
        elif "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"')

            # Try to convert to number
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass

            if current_section in result:
                result[current_section][key] = val
            else:
                result[key] = val

    return result


# =============================================================================
# External Metadata File Loaders
# =============================================================================


def load_profile_file(path: str | Path) -> dict[str, Any]:
    """
    Load a Tomocube .prf profile file (INI format).

    Profile files contain processing algorithm parameters, calibration data,
    and NA-specific optimization settings.

    Args:
        path: Path to the .prf file

    Returns:
        Dictionary with section names as keys, containing key-value pairs.
        Values are automatically converted to int/float when possible.

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read

    Example:
        >>> profile = load_profile_file("Cell.img.prf")
        >>> print(profile["DefaultNAOption"]["DefaultStep"])
        20
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 as fallback for older files
        content = path.read_text(encoding="latin-1")

    return parse_ini_string(content)


def load_vessel_file(path: str | Path) -> dict[str, Any]:
    """
    Load a Tomocube .vessel file (JSON format).

    Vessel files contain well plate geometry, coordinates, and vessel model
    specifications including physical dimensions and well positions.

    Args:
        path: Path to the .vessel file

    Returns:
        Dictionary containing vessel configuration with keys like:
        - Model: Vessel model name (e.g., "Ibidi μ-Slide 8well")
        - Width, Height: Physical dimensions in mm
        - WellRows, WellColumns: Grid layout
        - Wells: List of well definitions with positions

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON

    Example:
        >>> vessel = load_vessel_file(".vessel")
        >>> print(vessel["Model"])
        "Ibidi μ-Slide 8well"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vessel file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def load_project_file(path: str | Path) -> dict[str, Any]:
    """
    Load a Tomocube .tcxpro project file (JSON format).

    Project files contain top-level project metadata including
    project title and description.

    Args:
        path: Path to the .tcxpro file

    Returns:
        Dictionary containing project metadata with keys like:
        - Title: Project title
        - Description: Project description

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON

    Example:
        >>> project = load_project_file("2052_lab.tcxpro")
        >>> print(project["Title"])
        "Lab 2052 Project"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def load_experiment_file(path: str | Path) -> dict[str, Any]:
    """
    Load a Tomocube .tcxexp or .experiment file (JSON format).

    Experiment files contain experiment-level configuration including
    imaging scenarios, sequences, conditions, and acquisition parameters.

    Args:
        path: Path to the .tcxexp or .experiment file

    Returns:
        Dictionary containing experiment configuration with keys like:
        - Created: Creation timestamp
        - ImagingScenario: Imaging configuration
        - ImagingLocations: List of acquisition locations
        - Medium: Sample medium info (name, RI)
        - SampleType, VesselModel: Sample and vessel info

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON

    Example:
        >>> exp = load_experiment_file(".experiment")
        >>> print(exp["Medium"]["RI"])
        1.337
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def discover_related_metadata(tcf_path: str | Path) -> dict[str, Any]:
    """
    Discover and load all metadata files related to a TCF file.

    Automatically finds and loads:
    - .vessel file in same directory (well plate geometry)
    - .experiment file in same directory (experiment configuration)
    - .prf files in parent profile/ directory (processing profiles)

    Args:
        tcf_path: Path to a TCF file

    Returns:
        Dictionary with keys:
        - "vessel": Vessel metadata (if found)
        - "experiment": Experiment metadata (if found)
        - "profiles": Dict of profile name -> profile data (if found)

    Example:
        >>> meta = discover_related_metadata("data/session/file.TCF")
        >>> if "experiment" in meta:
        ...     print(meta["experiment"]["medium"]["mediumRI"])
        1.337
    """
    tcf_path = Path(tcf_path)
    session_dir = tcf_path.parent
    result: dict[str, Any] = {}

    # Look for .vessel file in session directory
    vessel_path = session_dir / ".vessel"
    if vessel_path.exists():
        try:
            result["vessel"] = load_vessel_file(vessel_path)
        except Exception as e:
            logger.debug(f"Failed to load vessel file: {e}")

    # Look for .experiment file in session directory
    experiment_path = session_dir / ".experiment"
    if experiment_path.exists():
        try:
            result["experiment"] = load_experiment_file(experiment_path)
        except Exception as e:
            logger.debug(f"Failed to load experiment file: {e}")

    # Look for profile files in session's profile/ subdirectory
    # Structure: session_dir/file.TCF, session_dir/profile/*.prf
    profile_dir = session_dir / "profile"
    if profile_dir.exists():
        profiles: dict[str, Any] = {}
        for prf_file in profile_dir.glob("*.prf"):
            try:
                # Use stem without "Cell." prefix for cleaner names
                name = prf_file.stem
                if name.startswith("Cell."):
                    name = name[5:]  # Remove "Cell." prefix
                profiles[name] = load_profile_file(prf_file)
            except Exception as e:
                logger.debug(f"Failed to load profile {prf_file}: {e}")
        if profiles:
            result["profiles"] = profiles

    return result
