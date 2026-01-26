"""
Tomocube Tools CLI entry point.

Usage:
    python -m tomocube view <file.TCF>              View a TCF file interactively
    python -m tomocube slice <file.TCF>             Compare HT/FL slices side-by-side
    python -m tomocube info <file.TCF>              Show file information
    python -m tomocube tiff <file.TCF> [output]     Convert to TIFF stack
    python -m tomocube mat <file.TCF> [output]      Convert to MATLAB .mat
    python -m tomocube gif <file.TCF> [output]      Create animated GIF
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


def main() -> int:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("Commands: view, slice, info, tiff, mat, gif")
        return 1

    command = sys.argv[1].lower()
    # Join remaining args to handle paths with spaces
    file_path = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    if command == "view":
        from tomocube.viewer import TCFViewer
        if not file_path:
            from tomocube.viewer.tcf_viewer import main as viewer_main
            viewer_main()
            return 0
        with TCFViewer(file_path) as viewer:
            viewer.show()
        return 0

    elif command == "slice":
        from tomocube.viewer import SliceViewer
        if not file_path:
            from tomocube.viewer.slice_viewer import main as slice_main
            slice_main()
            return 0
        viewer = SliceViewer(file_path)
        viewer.show()
        return 0

    elif command == "info":
        from tomocube.core import TCFFile
        if not file_path:
            print("Usage: python -m tomocube info <file.TCF>")
            return 1
        tcf_path = Path(file_path)
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

    elif command == "tiff":
        return _convert_tiff(file_path)

    elif command == "mat":
        return _convert_mat(file_path)

    elif command == "gif":
        return _convert_gif(file_path)

    else:
        print(f"Unknown command: {command}")
        print("Commands: view, slice, info, tiff, mat, gif")
        return 1


def _convert_tiff(args: str) -> int:
    """Convert TCF to TIFF stack."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_tiff

    parts = args.split() if args else []
    if not parts:
        print("Usage: python -m tomocube tiff <file.TCF> [output.tiff] [--fl CH0] [--16bit|--32bit]")
        return 1

    tcf_path = parts[0]
    output_path = None
    channel = "ht"
    bit_depth = 16

    i = 1
    while i < len(parts):
        if parts[i] == "--fl" and i + 1 < len(parts):
            channel = parts[i + 1]
            i += 2
        elif parts[i] == "--16bit":
            bit_depth = 16
            i += 1
        elif parts[i] == "--32bit":
            bit_depth = 32
            i += 1
        elif not output_path:
            output_path = parts[i]
            i += 1
        else:
            i += 1

    if not output_path:
        output_path = Path(tcf_path).stem + f"_{channel}.tiff"

    print(f"Converting {tcf_path} to TIFF...")
    print(f"  Channel: {channel}")
    print(f"  Bit depth: {bit_depth}")

    with TCFFileLoader(tcf_path) as loader:
        loader.load_timepoint(0)
        result = export_to_tiff(loader, output_path, channel=channel, bit_depth=bit_depth)

    print(f"Saved: {result}")
    return 0


def _convert_mat(args: str) -> int:
    """Convert TCF to MATLAB .mat format."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_mat

    parts = args.split() if args else []
    if not parts:
        print("Usage: python -m tomocube mat <file.TCF> [output.mat] [--no-fl]")
        return 1

    tcf_path = parts[0]
    output_path = None
    include_fl = True

    i = 1
    while i < len(parts):
        if parts[i] == "--no-fl":
            include_fl = False
            i += 1
        elif not output_path:
            output_path = parts[i]
            i += 1
        else:
            i += 1

    if not output_path:
        output_path = Path(tcf_path).stem + ".mat"

    print(f"Converting {tcf_path} to MAT...")
    print(f"  Include FL: {include_fl}")

    with TCFFileLoader(tcf_path) as loader:
        loader.load_timepoint(0)
        result = export_to_mat(loader, output_path, include_fl=include_fl)

    print(f"Saved: {result}")
    return 0


def _convert_gif(args: str) -> int:
    """Convert TCF to animated GIF."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_overlay_gif, export_to_gif

    parts = args.split() if args else []
    if not parts:
        print("Usage: python -m tomocube gif <file.TCF> [output.gif] [--overlay] [--fps N] [--axis z|y|x]")
        return 1

    tcf_path = parts[0]
    output_path = None
    overlay = False
    fps = 10
    axis = "z"

    i = 1
    while i < len(parts):
        if parts[i] == "--overlay":
            overlay = True
            i += 1
        elif parts[i] == "--fps" and i + 1 < len(parts):
            fps = int(parts[i + 1])
            i += 2
        elif parts[i] == "--axis" and i + 1 < len(parts):
            axis = parts[i + 1]
            i += 2
        elif not output_path:
            output_path = parts[i]
            i += 1
        else:
            i += 1

    if not output_path:
        suffix = "_overlay" if overlay else f"_{axis}"
        output_path = Path(tcf_path).stem + suffix + ".gif"

    print(f"Converting {tcf_path} to GIF...")
    print(f"  Mode: {'overlay' if overlay else 'single channel'}")
    print(f"  FPS: {fps}")
    if not overlay:
        print(f"  Axis: {axis}")

    with TCFFileLoader(tcf_path) as loader:
        loader.load_timepoint(0)
        if overlay:
            if not loader.has_fluorescence:
                print("Error: No fluorescence data for overlay mode")
                return 1
            result = export_overlay_gif(loader, output_path, fps=fps)
        else:
            result = export_to_gif(loader, output_path, axis=axis, fps=fps)

    print(f"Saved: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
