"""
TCF Metadata - Metadata extraction and parsing utilities.

This module provides functions for extracting and parsing
metadata from TCF files.
"""

from __future__ import annotations

import json
import logging
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
