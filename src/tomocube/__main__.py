"""
Tomocube Tools CLI entry point.

Usage:
    python -m tomocube view <file.TCF>      View a TCF file interactively
    python -m tomocube slice <file.TCF>     Compare HT/FL slices side-by-side
    python -m tomocube info <file.TCF>      Show file information
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


def main() -> int:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("Commands: view, slice, info")
        return 1

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "view":
        from tomocube.viewer import TCFViewer
        if not args:
            # Try to find a file automatically
            from tomocube.viewer.tcf_viewer import main as viewer_main
            viewer_main()
            return 0
        with TCFViewer(args[0]) as viewer:
            viewer.show()
        return 0

    elif command == "slice":
        from tomocube.viewer import SliceViewer
        if not args:
            # Try to find a file automatically
            from tomocube.viewer.slice_viewer import main as slice_main
            slice_main()
            return 0
        viewer = SliceViewer(args[0])
        viewer.show()
        return 0

    elif command == "info":
        from tomocube.core import TCFFile
        if not args:
            print("Usage: python -m tomocube info <file.TCF>")
            return 1
        tcf_path = Path(args[0])
        with h5py.File(tcf_path, "r") as f:
            info = TCFFile.from_hdf5(f)
        print(f"File: {tcf_path.name}")
        print(f"HT Shape: {info.ht_shape} (Z, Y, X)")
        print(f"HT Resolution: {info.ht_resolution} um/px (Z, Y, X)")
        print(f"Magnification: {info.magnification}x")
        print(f"NA: {info.numerical_aperture}")
        print(f"Medium RI: {info.medium_ri}")
        print(f"Timepoints: {len(info.timepoints)}")
        print(f"Has Fluorescence: {info.has_fluorescence}")
        if info.has_fluorescence:
            print(f"FL Channels: {info.fl_channels}")
            print(f"FL Shapes: {info.fl_shapes}")
            print(f"FL Resolution: {info.fl_resolution} um/px (Z, Y, X)")
        if info.ri_min is not None and info.ri_max is not None:
            print(f"RI Range: [{info.ri_min:.4f}, {info.ri_max:.4f}]")
        return 0

    else:
        print(f"Unknown command: {command}")
        print("Commands: view, slice, info")
        return 1


if __name__ == "__main__":
    sys.exit(main())
