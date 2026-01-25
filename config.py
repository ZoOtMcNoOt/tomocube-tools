"""
Configuration module for tomocube-tools.

Handles environment-based configuration for test data paths and output directories.

Environment Variables:
    TCF_TEST_DATA_DIR   Base directory containing test TCF files
    TCF_OUTPUT_DIR      Directory for output files (default: ./output)

Usage:
    from config import get_test_data_dir, get_output_dir, find_test_file
    
    # Get configured directories
    data_dir = get_test_data_dir()
    output_dir = get_output_dir()
    
    # Find a specific test file
    tcf_path = find_test_file("260114.123923.melanoma b16.001")
"""

from __future__ import annotations

import os
from pathlib import Path


def get_test_data_dir() -> Path:
    """
    Get the test data directory from environment or default location.
    
    Checks:
        1. TCF_TEST_DATA_DIR environment variable
        2. ./data directory relative to module
        3. ~/tomocube-data if it exists
    
    Returns:
        Path to test data directory, or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get("TCF_TEST_DATA_DIR")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    
    # Check relative ./data directory
    module_dir = Path(__file__).parent
    local_data = module_dir / "data"
    if local_data.exists():
        return local_data
    
    # Check home directory
    home_data = Path.home() / "tomocube-data"
    if home_data.exists():
        return home_data
    
    # Return default (may not exist)
    return local_data


def get_output_dir() -> Path:
    """
    Get the output directory from environment or default location.
    
    Checks:
        1. TCF_OUTPUT_DIR environment variable
        2. ./output directory relative to module (created if needed)
    
    Returns:
        Path to output directory
    """
    env_path = os.environ.get("TCF_OUTPUT_DIR")
    if env_path:
        path = Path(env_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Default to ./output
    module_dir = Path(__file__).parent
    output_dir = module_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_tcf_files(directory: str | Path | None = None) -> list[Path]:
    """
    Find all TCF files in a directory recursively.
    
    Args:
        directory: Directory to search (default: test data directory)
    
    Returns:
        List of paths to TCF files, sorted by name
    """
    if directory is None:
        directory = get_test_data_dir()
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return sorted(directory.rglob("*.TCF"))


def find_test_file(pattern: str) -> Path | None:
    """
    Find a test file matching the given pattern.
    
    Args:
        pattern: Substring to match in filename (e.g., "S001" or "melanoma b16.001")
    
    Returns:
        Path to matching file, or None if not found
    """
    tcf_files = find_tcf_files()
    
    for tcf_path in tcf_files:
        if pattern in tcf_path.name or pattern in str(tcf_path):
            return tcf_path
    
    return None


def get_default_test_file() -> Path | None:
    """
    Get the first available test file.
    
    Returns:
        Path to first TCF file found, or None
    """
    tcf_files = find_tcf_files()
    return tcf_files[0] if tcf_files else None


# Default test file patterns for different scripts
DEFAULT_TEST_PATTERNS = {
    "tcf_viewer": "S001",        # Any sample works
    "slice_viewer": "S008",      # Sample with good FL data
    "fl_registration": "S010",   # Sample with FL for registration testing
}


def get_script_default_file(script_name: str) -> Path | None:
    """
    Get the default test file for a specific script.
    
    Args:
        script_name: Name of the script (tcf_viewer, slice_viewer, fl_registration)
    
    Returns:
        Path to appropriate test file, or first available file
    """
    pattern = DEFAULT_TEST_PATTERNS.get(script_name)
    if pattern:
        result = find_test_file(pattern)
        if result:
            return result
    
    # Fallback to first available
    return get_default_test_file()
