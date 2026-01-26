"""
Tomocube Tools CLI - Work with Tomocube TCF holotomography files.

Run 'python -m tomocube help' for usage information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

# ANSI color codes for terminal output
class Colors:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.BOLD = cls.DIM = cls.CYAN = cls.GREEN = ""
        cls.YELLOW = cls.BLUE = cls.MAGENTA = cls.RED = cls.RESET = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def _print_banner() -> None:
    """Print the CLI banner."""
    c = Colors
    print(f"""
{c.CYAN}{c.BOLD}  +--------------------------------------------+
  |           TOMOCUBE TOOLS v0.1.0            |
  |     Holotomography Data Processing CLI     |
  +--------------------------------------------+{c.RESET}
""")


def _print_help() -> None:
    """Print comprehensive help information."""
    c = Colors
    _print_banner()

    print(f"""{c.BOLD}USAGE{c.RESET}
    python -m tomocube {c.CYAN}<command>{c.RESET} {c.DIM}[options]{c.RESET}

{c.BOLD}COMMANDS{c.RESET}

  {c.GREEN}Visualization{c.RESET}
    {c.CYAN}view{c.RESET}   <file.TCF>              Interactive 3D viewer with FL overlay
    {c.CYAN}slice{c.RESET}  <file.TCF>              Side-by-side HT/FL comparison

  {c.GREEN}Information{c.RESET}
    {c.CYAN}info{c.RESET}   <file.TCF>              Display file metadata and structure
    {c.CYAN}help{c.RESET}                           Show this help message

  {c.GREEN}Export{c.RESET}
    {c.CYAN}tiff{c.RESET}   <file.TCF> [output]     Export to multi-page TIFF stack
    {c.CYAN}mat{c.RESET}    <file.TCF> [output]     Export to MATLAB .mat format
    {c.CYAN}gif{c.RESET}    <file.TCF> [output]     Create animated GIF

{c.BOLD}EXPORT OPTIONS{c.RESET}

  {c.YELLOW}tiff{c.RESET} options:
    --fl <channel>    Export fluorescence channel (e.g., CH0) instead of HT
    --16bit           16-bit output {c.DIM}(default){c.RESET}
    --32bit           32-bit float output

  {c.YELLOW}mat{c.RESET} options:
    --no-fl           Exclude fluorescence data from export

  {c.YELLOW}gif{c.RESET} options:
    --overlay         Create HT+FL overlay animation
    --fps <N>         Frame rate {c.DIM}(default: 10){c.RESET}
    --axis <z|y|x>    Slice axis for animation {c.DIM}(default: z){c.RESET}

{c.BOLD}VIEWER KEYBOARD SHORTCUTS{c.RESET}

  {c.GREEN}Navigation{c.RESET}
    Up/Down, Scroll   Navigate Z-slices
    Home/End          Jump to first/last slice
    Click             Select position in any view

  {c.GREEN}Display{c.RESET}
    A                 Auto-contrast (current slice)
    G                 Auto-contrast (global)
    R                 Reset view
    I                 Invert colormap
    1-6               Switch colormap
    F                 Toggle fluorescence overlay

  {c.GREEN}Measurement{c.RESET}
    D                 Distance measurement mode
    P                 Polygon/area measurement mode
    C                 Clear all measurements

  {c.GREEN}Export{c.RESET}
    S                 Save current slice as PNG
    M                 Save MIP as PNG

  {c.GREEN}Exit{c.RESET}
    Q, Escape         Quit viewer

{c.BOLD}EXAMPLES{c.RESET}

  {c.DIM}# View a TCF file interactively{c.RESET}
  python -m tomocube view "path/to/sample.TCF"

  {c.DIM}# Show file information{c.RESET}
  python -m tomocube info "path/to/sample.TCF"

  {c.DIM}# Export HT data to TIFF{c.RESET}
  python -m tomocube tiff "sample.TCF" output.tiff

  {c.DIM}# Export FL channel to 32-bit TIFF{c.RESET}
  python -m tomocube tiff "sample.TCF" fl.tiff --fl CH0 --32bit

  {c.DIM}# Create Z-stack animation{c.RESET}
  python -m tomocube gif "sample.TCF" animation.gif --fps 15

  {c.DIM}# Create HT+FL overlay animation{c.RESET}
  python -m tomocube gif "sample.TCF" overlay.gif --overlay

{c.BOLD}MORE INFORMATION{c.RESET}

  Documentation:  See README.md
  Data format:    See DATA_ANALYSIS.md
""")


def _print_short_help() -> None:
    """Print short help when no command given."""
    c = Colors
    _print_banner()

    print(f"""{c.BOLD}USAGE{c.RESET}
    python -m tomocube {c.CYAN}<command>{c.RESET} {c.DIM}[options]{c.RESET}

{c.BOLD}COMMANDS{c.RESET}
    {c.CYAN}view{c.RESET}    Interactive 3D viewer         {c.CYAN}info{c.RESET}    Show file metadata
    {c.CYAN}slice{c.RESET}   HT/FL comparison viewer       {c.CYAN}help{c.RESET}    Full help & examples
    {c.CYAN}tiff{c.RESET}    Export to TIFF stack          {c.CYAN}mat{c.RESET}     Export to MATLAB
    {c.CYAN}gif{c.RESET}     Create animated GIF

{c.DIM}Run 'python -m tomocube help' for detailed usage and examples.{c.RESET}
""")


def _print_info(file_path: str) -> int:
    """Print formatted file information."""
    from tomocube.core import TCFFile

    c = Colors
    tcf_path = Path(file_path)

    if not tcf_path.exists():
        print(f"{c.RED}Error:{c.RESET} File not found: {file_path}")
        return 1

    try:
        with h5py.File(tcf_path, "r") as f:
            info = TCFFile.from_hdf5(f)
    except Exception as e:
        print(f"{c.RED}Error:{c.RESET} Could not read file: {e}")
        return 1

    # Calculate physical dimensions
    ht_z, ht_y, ht_x = info.ht_shape
    res_z, res_y, res_x = info.ht_resolution
    fov_x = ht_x * res_x
    fov_y = ht_y * res_y
    fov_z = ht_z * res_z

    print(f"""
{c.CYAN}{c.BOLD}+--------------------------------------------------------------+
|  TCF FILE INFORMATION                                        |
+--------------------------------------------------------------+{c.RESET}

{c.BOLD}File{c.RESET}
    Name:           {c.GREEN}{tcf_path.name}{c.RESET}
    Path:           {c.DIM}{tcf_path.parent}{c.RESET}

{c.BOLD}Holotomography (HT){c.RESET}
    Shape:          {c.YELLOW}{ht_z}{c.RESET} x {c.YELLOW}{ht_y}{c.RESET} x {c.YELLOW}{ht_x}{c.RESET} {c.DIM}(Z x Y x X){c.RESET}
    Resolution:     {res_x:.3f} x {res_y:.3f} x {res_z:.3f} um/px {c.DIM}(X x Y x Z){c.RESET}
    Field of View:  {fov_x:.1f} x {fov_y:.1f} x {fov_z:.1f} um""")

    if info.ri_min is not None and info.ri_max is not None:
        print(f"    RI Range:       {c.CYAN}{info.ri_min:.4f}{c.RESET} - {c.CYAN}{info.ri_max:.4f}{c.RESET}")

    print(f"""
{c.BOLD}Optics{c.RESET}
    Magnification:  {c.YELLOW}{info.magnification or '?'}x{c.RESET}
    NA:             {info.numerical_aperture or '?'}
    Medium RI:      {info.medium_ri or '?'}

{c.BOLD}Acquisition{c.RESET}
    Timepoints:     {len(info.timepoints)}""")

    if info.has_fluorescence:
        print(f"""
{c.BOLD}Fluorescence (FL){c.RESET}  {c.GREEN}Available{c.RESET}
    Channels:       {', '.join(info.fl_channels)}""")
        for ch, shape in info.fl_shapes.items():
            fl_z, fl_y, fl_x = shape
            fl_res = info.fl_resolution
            fl_fov_x = fl_x * fl_res[2]
            fl_fov_y = fl_y * fl_res[1]
            fl_fov_z = fl_z * fl_res[0]
            print(f"    {ch} Shape:       {fl_z} x {fl_y} x {fl_x} {c.DIM}(Z x Y x X){c.RESET}")
            print(f"    {ch} FOV:         {fl_fov_x:.1f} x {fl_fov_y:.1f} x {fl_fov_z:.1f} um")
        print(f"    Resolution:     {fl_res[2]:.3f} x {fl_res[1]:.3f} x {fl_res[0]:.3f} um/px {c.DIM}(X x Y x Z){c.RESET}")
    else:
        print(f"""
{c.BOLD}Fluorescence (FL){c.RESET}  {c.DIM}Not available{c.RESET}""")

    print()
    return 0


def _print_progress(message: str, done: bool = False) -> None:
    """Print a progress message."""
    c = Colors
    if done:
        print(f"  {c.GREEN}[OK]{c.RESET} {message}")
    else:
        print(f"  {c.YELLOW}>>>{c.RESET} {message}")


def _print_error(message: str) -> None:
    """Print an error message."""
    c = Colors
    print(f"\n{c.RED}Error:{c.RESET} {message}\n")


def _print_success(message: str) -> None:
    """Print a success message."""
    c = Colors
    print(f"\n{c.GREEN}SUCCESS:{c.RESET} {message}\n")


def main() -> int:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        _print_short_help()
        return 1

    command = sys.argv[1].lower()

    # Help command
    if command in ("help", "-h", "--help", "-?"):
        _print_help()
        return 0

    # Version command
    if command in ("version", "-v", "--version"):
        print("tomocube-tools v0.1.0")
        return 0

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
        if not file_path:
            _print_error("Missing file path")
            print("Usage: python -m tomocube info <file.TCF>")
            return 1
        return _print_info(file_path)

    elif command == "tiff":
        return _convert_tiff(file_path)

    elif command == "mat":
        return _convert_mat(file_path)

    elif command == "gif":
        return _convert_gif(file_path)

    else:
        _print_error(f"Unknown command: {command}")
        _print_short_help()
        return 1


def _convert_tiff(args: str) -> int:
    """Convert TCF to TIFF stack."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_tiff

    c = Colors
    parts = args.split() if args else []

    if not parts:
        print(f"""
{c.BOLD}TIFF Export{c.RESET}

{c.BOLD}Usage:{c.RESET}
    python -m tomocube tiff <file.TCF> [output.tiff] [options]

{c.BOLD}Options:{c.RESET}
    --fl <channel>    Export fluorescence channel (e.g., CH0)
    --16bit           16-bit unsigned integer output {c.DIM}(default){c.RESET}
    --32bit           32-bit float output

{c.BOLD}Examples:{c.RESET}
    python -m tomocube tiff sample.TCF
    python -m tomocube tiff sample.TCF output.tiff --32bit
    python -m tomocube tiff sample.TCF fl_stack.tiff --fl CH0
""")
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

    if not Path(tcf_path).exists():
        _print_error(f"File not found: {tcf_path}")
        return 1

    print(f"\n{c.BOLD}Exporting to TIFF{c.RESET}")
    _print_progress(f"Input: {tcf_path}")
    _print_progress(f"Channel: {channel}")
    _print_progress(f"Bit depth: {bit_depth}")
    _print_progress("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)
            _print_progress("Converting...", done=True)
            result = export_to_tiff(loader, output_path, channel=channel, bit_depth=bit_depth)

        _print_success(f"Saved: {result}")
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


def _convert_mat(args: str) -> int:
    """Convert TCF to MATLAB .mat format."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_mat

    c = Colors
    parts = args.split() if args else []

    if not parts:
        print(f"""
{c.BOLD}MATLAB Export{c.RESET}

{c.BOLD}Usage:{c.RESET}
    python -m tomocube mat <file.TCF> [output.mat] [options]

{c.BOLD}Options:{c.RESET}
    --no-fl           Exclude fluorescence data

{c.BOLD}Output Variables:{c.RESET}
    ht_data           3D HT volume (Z x Y x X)
    fl_CH0, etc.      FL channel volumes (if present)
    ht_resolution     [Z, Y, X] resolution in um
    fl_resolution     [Z, Y, X] FL resolution in um
    magnification     Objective magnification
    numerical_aperture
    medium_ri         Medium refractive index

{c.BOLD}Examples:{c.RESET}
    python -m tomocube mat sample.TCF
    python -m tomocube mat sample.TCF output.mat --no-fl
""")
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

    if not Path(tcf_path).exists():
        _print_error(f"File not found: {tcf_path}")
        return 1

    print(f"\n{c.BOLD}Exporting to MATLAB{c.RESET}")
    _print_progress(f"Input: {tcf_path}")
    _print_progress(f"Include FL: {include_fl}")
    _print_progress("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)
            _print_progress("Converting...", done=True)
            result = export_to_mat(loader, output_path, include_fl=include_fl)

        _print_success(f"Saved: {result}")
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


def _convert_gif(args: str) -> int:
    """Convert TCF to animated GIF."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_overlay_gif, export_to_gif

    c = Colors
    parts = args.split() if args else []

    if not parts:
        print(f"""
{c.BOLD}GIF Animation{c.RESET}

{c.BOLD}Usage:{c.RESET}
    python -m tomocube gif <file.TCF> [output.gif] [options]

{c.BOLD}Options:{c.RESET}
    --overlay         Create HT+FL overlay animation (green FL on grayscale HT)
    --fps <N>         Frame rate {c.DIM}(default: 10){c.RESET}
    --axis <z|y|x>    Slice axis for animation {c.DIM}(default: z){c.RESET}

{c.BOLD}Examples:{c.RESET}
    python -m tomocube gif sample.TCF                    {c.DIM}# Z-stack animation{c.RESET}
    python -m tomocube gif sample.TCF anim.gif --fps 15  {c.DIM}# Custom frame rate{c.RESET}
    python -m tomocube gif sample.TCF --axis y           {c.DIM}# Y-slice animation{c.RESET}
    python -m tomocube gif sample.TCF --overlay          {c.DIM}# HT+FL overlay{c.RESET}
""")
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

    if not Path(tcf_path).exists():
        _print_error(f"File not found: {tcf_path}")
        return 1

    print(f"\n{c.BOLD}Creating GIF Animation{c.RESET}")
    _print_progress(f"Input: {tcf_path}")
    _print_progress(f"Mode: {'HT+FL overlay' if overlay else f'{axis.upper()}-stack'}")
    _print_progress(f"Frame rate: {fps} fps")
    _print_progress("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)

            if overlay:
                if not loader.has_fluorescence:
                    _print_error("No fluorescence data available for overlay mode")
                    return 1
                _print_progress("Generating overlay frames...", done=True)
                result = export_overlay_gif(loader, output_path, fps=fps)
            else:
                _print_progress(f"Generating {axis.upper()}-slice frames...", done=True)
                result = export_to_gif(loader, output_path, axis=axis, fps=fps)

        _print_success(f"Saved: {result}")
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
