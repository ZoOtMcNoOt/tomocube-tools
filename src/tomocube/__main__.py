"""
Tomocube Tools CLI - Work with Tomocube TCF holotomography files.

Run 'python -m tomocube help' for usage information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


class Style:
    """Terminal styling with ANSI codes."""

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"

    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable all styling."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


# Disable colors if not a TTY or on Windows without ANSI support
if not sys.stdout.isatty():
    Style.disable()


def _styled(text: str, *styles: str) -> str:
    """Apply multiple styles to text."""
    return "".join(styles) + text + Style.RESET


def _print_logo() -> None:
    """Print the Tomocube Tools logo."""
    s = Style
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}TOMOCUBE{s.RESET} {s.DIM}Tools{s.RESET} {s.BRIGHT_BLACK}v0.1.0{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}Holotomography data processing{s.RESET}")
    print()


def _print_help() -> None:
    """Print comprehensive help information."""
    s = Style

    _print_logo()

    # Usage
    print(f"  {s.BOLD}Usage:{s.RESET} python -m tomocube {s.CYAN}<command>{s.RESET} {s.DIM}[options]{s.RESET}")
    print()

    # Commands section
    print(f"  {s.BOLD}Commands{s.RESET}")
    print()

    # Visualization
    print(f"    {s.BRIGHT_BLACK}Visualization{s.RESET}")
    print(f"      {s.CYAN}view{s.RESET}  {s.DIM}<file>{s.RESET}    Interactive 3D viewer with orthogonal slices")
    print(f"      {s.CYAN}slice{s.RESET} {s.DIM}<file>{s.RESET}    Side-by-side HT/FL comparison viewer")
    print()

    # Information
    print(f"    {s.BRIGHT_BLACK}Information{s.RESET}")
    print(f"      {s.CYAN}info{s.RESET}  {s.DIM}<file>{s.RESET}    Display file metadata and structure")
    print(f"      {s.CYAN}help{s.RESET}            Show this help message")
    print()

    # Export
    print(f"    {s.BRIGHT_BLACK}Export{s.RESET}")
    print(f"      {s.CYAN}tiff{s.RESET}  {s.DIM}<file>{s.RESET}    Export to multi-page TIFF stack")
    print(f"      {s.CYAN}mat{s.RESET}   {s.DIM}<file>{s.RESET}    Export to MATLAB .mat format")
    print(f"      {s.CYAN}gif{s.RESET}   {s.DIM}<file>{s.RESET}    Create animated GIF")
    print()

    # Options
    print(f"  {s.BOLD}Export Options{s.RESET}")
    print()
    print(f"    {s.YELLOW}tiff{s.RESET}")
    print(f"      {s.DIM}--fl <CH>{s.RESET}       Export fluorescence channel instead of HT")
    print(f"      {s.DIM}--16bit{s.RESET}         16-bit output {s.BRIGHT_BLACK}(default){s.RESET}")
    print(f"      {s.DIM}--32bit{s.RESET}         32-bit float output")
    print()
    print(f"    {s.YELLOW}mat{s.RESET}")
    print(f"      {s.DIM}--no-fl{s.RESET}         Exclude fluorescence data")
    print()
    print(f"    {s.YELLOW}gif{s.RESET}")
    print(f"      {s.DIM}--overlay{s.RESET}       HT+FL overlay animation")
    print(f"      {s.DIM}--fps <N>{s.RESET}       Frame rate {s.BRIGHT_BLACK}(default: 10){s.RESET}")
    print(f"      {s.DIM}--axis <z|y|x>{s.RESET}  Slice axis {s.BRIGHT_BLACK}(default: z){s.RESET}")
    print()

    # Keyboard shortcuts
    print(f"  {s.BOLD}Viewer Shortcuts{s.RESET}")
    print()
    _print_shortcut_row([
        ("Up/Down", "Navigate Z"),
        ("A", "Auto contrast"),
        ("D", "Distance"),
        ("S", "Save PNG"),
    ])
    _print_shortcut_row([
        ("Scroll", "Navigate Z"),
        ("G", "Global contrast"),
        ("P", "Polygon area"),
        ("M", "Save MIP"),
    ])
    _print_shortcut_row([
        ("Click", "Select pos"),
        ("I", "Invert cmap"),
        ("C", "Clear meas"),
        ("F", "Toggle FL"),
    ])
    _print_shortcut_row([
        ("1-6", "Colormap"),
        ("R", "Reset view"),
        ("Q", "Quit"),
        ("", ""),
    ])
    print()

    # Examples
    print(f"  {s.BOLD}Examples{s.RESET}")
    print()
    print(f"    {s.BRIGHT_BLACK}# View a TCF file{s.RESET}")
    print(f"    {s.DIM}${s.RESET} python -m tomocube view sample.TCF")
    print()
    print(f"    {s.BRIGHT_BLACK}# Export to TIFF{s.RESET}")
    print(f"    {s.DIM}${s.RESET} python -m tomocube tiff sample.TCF --32bit")
    print()
    print(f"    {s.BRIGHT_BLACK}# Create overlay animation{s.RESET}")
    print(f"    {s.DIM}${s.RESET} python -m tomocube gif sample.TCF --overlay --fps 15")
    print()


def _print_shortcut_row(shortcuts: list[tuple[str, str]]) -> None:
    """Print a row of keyboard shortcuts."""
    s = Style
    parts = []
    for key, desc in shortcuts:
        if key:
            parts.append(f"{s.CYAN}{key:12}{s.RESET}{s.DIM}{desc:14}{s.RESET}")
        else:
            parts.append(" " * 26)
    print(f"    {''.join(parts)}")


def _print_short_usage() -> None:
    """Print short usage when no command given."""
    s = Style
    _print_logo()

    print(f"  {s.BOLD}Usage:{s.RESET} python -m tomocube {s.CYAN}<command>{s.RESET} {s.DIM}[options]{s.RESET}")
    print()
    print(f"  {s.BOLD}Commands{s.RESET}")
    print(f"    {s.CYAN}view{s.RESET}   Interactive 3D viewer      {s.CYAN}tiff{s.RESET}   Export TIFF stack")
    print(f"    {s.CYAN}slice{s.RESET}  HT/FL comparison           {s.CYAN}mat{s.RESET}    Export MATLAB .mat")
    print(f"    {s.CYAN}info{s.RESET}   Show file metadata         {s.CYAN}gif{s.RESET}    Create animation")
    print()
    print(f"  {s.DIM}Run{s.RESET} python -m tomocube help {s.DIM}for detailed usage{s.RESET}")
    print()


def _print_info(file_path: str) -> int:
    """Print formatted file information."""
    from tomocube.core import TCFFile

    s = Style
    tcf_path = Path(file_path)

    if not tcf_path.exists():
        _print_error(f"File not found: {file_path}")
        return 1

    try:
        with h5py.File(tcf_path, "r") as f:
            info = TCFFile.from_hdf5(f)
    except Exception as e:
        _print_error(f"Could not read file: {e}")
        return 1

    # Calculate physical dimensions
    ht_z, ht_y, ht_x = info.ht_shape
    res_z, res_y, res_x = info.ht_resolution
    fov_x = ht_x * res_x
    fov_y = ht_y * res_y
    fov_z = ht_z * res_z

    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}TCF File Info{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{'=' * 50}{s.RESET}")
    print()

    # File
    print(f"  {s.BOLD}File{s.RESET}")
    print(f"    {s.DIM}Name{s.RESET}         {s.GREEN}{tcf_path.name}{s.RESET}")
    print(f"    {s.DIM}Location{s.RESET}     {s.BRIGHT_BLACK}{tcf_path.parent}{s.RESET}")
    print()

    # Holotomography
    print(f"  {s.BOLD}Holotomography{s.RESET}")
    print(f"    {s.DIM}Volume{s.RESET}       {s.YELLOW}{ht_z}{s.RESET} x {s.YELLOW}{ht_y}{s.RESET} x {s.YELLOW}{ht_x}{s.RESET} {s.BRIGHT_BLACK}(Z x Y x X){s.RESET}")
    print(f"    {s.DIM}Resolution{s.RESET}   {res_x:.3f} x {res_y:.3f} x {res_z:.3f} {s.BRIGHT_BLACK}um/px{s.RESET}")
    print(f"    {s.DIM}FOV{s.RESET}          {fov_x:.1f} x {fov_y:.1f} x {fov_z:.1f} {s.BRIGHT_BLACK}um{s.RESET}")
    if info.ri_min is not None and info.ri_max is not None:
        print(f"    {s.DIM}RI Range{s.RESET}     {s.CYAN}{info.ri_min:.4f}{s.RESET} - {s.CYAN}{info.ri_max:.4f}{s.RESET}")
    print()

    # Optics
    print(f"  {s.BOLD}Optics{s.RESET}")
    print(f"    {s.DIM}Magnification{s.RESET}  {s.YELLOW}{info.magnification or '?'}x{s.RESET}")
    print(f"    {s.DIM}NA{s.RESET}             {info.numerical_aperture or '?'}")
    print(f"    {s.DIM}Medium RI{s.RESET}      {info.medium_ri or '?'}")
    print()

    # Acquisition
    print(f"  {s.BOLD}Acquisition{s.RESET}")
    print(f"    {s.DIM}Timepoints{s.RESET}   {len(info.timepoints)}")
    print()

    # Fluorescence
    if info.has_fluorescence:
        print(f"  {s.BOLD}Fluorescence{s.RESET}  {s.GREEN}Available{s.RESET}")
        print(f"    {s.DIM}Channels{s.RESET}     {', '.join(info.fl_channels)}")
        for ch, shape in info.fl_shapes.items():
            fl_z, fl_y, fl_x = shape
            print(f"    {s.DIM}{ch} Volume{s.RESET}   {fl_z} x {fl_y} x {fl_x}")
        fl_res = info.fl_resolution
        print(f"    {s.DIM}Resolution{s.RESET}   {fl_res[2]:.3f} x {fl_res[1]:.3f} x {fl_res[0]:.3f} {s.BRIGHT_BLACK}um/px{s.RESET}")
    else:
        print(f"  {s.BOLD}Fluorescence{s.RESET}  {s.BRIGHT_BLACK}Not available{s.RESET}")

    print()
    return 0


def _print_error(message: str) -> None:
    """Print an error message."""
    s = Style
    print()
    print(f"  {s.RED}{s.BOLD}Error{s.RESET} {message}")
    print()


def _print_success(path: str) -> None:
    """Print a success message."""
    s = Style
    print(f"  {s.GREEN}{s.BOLD}Saved{s.RESET} {path}")
    print()


def _print_step(message: str, status: str = "working") -> None:
    """Print a step in a process."""
    s = Style
    if status == "working":
        print(f"  {s.YELLOW}>{s.RESET} {message}")
    elif status == "done":
        print(f"  {s.GREEN}>{s.RESET} {message}")
    elif status == "info":
        print(f"  {s.BRIGHT_BLACK}>{s.RESET} {s.DIM}{message}{s.RESET}")


def _print_export_header(title: str, input_file: str) -> None:
    """Print export operation header."""
    s = Style
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{title}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{'=' * 50}{s.RESET}")
    print()
    _print_step(f"Input: {Path(input_file).name}", "info")


def main() -> int:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        _print_short_usage()
        return 1

    command = sys.argv[1].lower()

    # Help command
    if command in ("help", "-h", "--help", "-?"):
        _print_help()
        return 0

    # Version command
    if command in ("version", "-v", "--version"):
        s = Style
        print(f"{s.BRIGHT_CYAN}tomocube-tools{s.RESET} {s.DIM}v0.1.0{s.RESET}")
        return 0

    # Get remaining args as list to preserve paths with spaces
    args_list = sys.argv[2:] if len(sys.argv) > 2 else []
    # For simple commands that just need a file path (no options)
    file_path = args_list[0] if args_list else ""

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
            print("  Usage: python -m tomocube info <file.TCF>")
            print()
            return 1
        return _print_info(file_path)

    elif command == "tiff":
        return _convert_tiff(args_list)

    elif command == "mat":
        return _convert_mat(args_list)

    elif command == "gif":
        return _convert_gif(args_list)

    else:
        _print_error(f"Unknown command: {command}")
        _print_short_usage()
        return 1


def _print_subcommand_help(name: str, usage: str, options: list[tuple[str, str, str]], examples: list[tuple[str, str]]) -> None:
    """Print help for a subcommand."""
    s = Style
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{name}{s.RESET}")
    print()
    print(f"  {s.BOLD}Usage{s.RESET}")
    print(f"    {s.DIM}${s.RESET} {usage}")
    print()

    if options:
        print(f"  {s.BOLD}Options{s.RESET}")
        for opt, arg, desc in options:
            if arg:
                print(f"    {s.CYAN}{opt}{s.RESET} {s.DIM}{arg}{s.RESET}")
                print(f"        {desc}")
            else:
                print(f"    {s.CYAN}{opt}{s.RESET}    {desc}")
        print()

    if examples:
        print(f"  {s.BOLD}Examples{s.RESET}")
        for comment, cmd in examples:
            print(f"    {s.BRIGHT_BLACK}# {comment}{s.RESET}")
            print(f"    {s.DIM}${s.RESET} {cmd}")
            print()


def _convert_tiff(args: list[str]) -> int:
    """Convert TCF to TIFF stack."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_tiff

    parts = args if args else []

    if not parts:
        _print_subcommand_help(
            "TIFF Export",
            "python -m tomocube tiff <file.TCF> [output.tiff] [options]",
            [
                ("--fl", "<channel>", "Export fluorescence channel (e.g., CH0)"),
                ("--16bit", "", "16-bit unsigned integer output (default)"),
                ("--32bit", "", "32-bit float output"),
            ],
            [
                ("Export HT data", "python -m tomocube tiff sample.TCF"),
                ("Export as 32-bit", "python -m tomocube tiff sample.TCF output.tiff --32bit"),
                ("Export FL channel", "python -m tomocube tiff sample.TCF fl.tiff --fl CH0"),
            ],
        )
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

    _print_export_header("TIFF Export", tcf_path)
    _print_step(f"Channel: {channel}", "info")
    _print_step(f"Bit depth: {bit_depth}", "info")
    print()
    _print_step("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)
            _print_step("Converting to TIFF...", "done")
            result = export_to_tiff(loader, output_path, channel=channel, bit_depth=bit_depth)

        _print_success(str(result))
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


def _convert_mat(args: list[str]) -> int:
    """Convert TCF to MATLAB .mat format."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_to_mat

    s = Style
    parts = args if args else []

    if not parts:
        _print_subcommand_help(
            "MATLAB Export",
            "python -m tomocube mat <file.TCF> [output.mat] [options]",
            [
                ("--no-fl", "", "Exclude fluorescence data"),
            ],
            [
                ("Export all data", "python -m tomocube mat sample.TCF"),
                ("Export HT only", "python -m tomocube mat sample.TCF output.mat --no-fl"),
            ],
        )
        print(f"  {s.BOLD}Output Variables{s.RESET}")
        print(f"    {s.CYAN}ht_data{s.RESET}           3D HT volume (Z x Y x X)")
        print(f"    {s.CYAN}fl_CH0{s.RESET}, etc.      FL channel volumes")
        print(f"    {s.CYAN}ht_resolution{s.RESET}     [Z, Y, X] resolution in um")
        print(f"    {s.CYAN}magnification{s.RESET}     Objective magnification")
        print()
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

    _print_export_header("MATLAB Export", tcf_path)
    _print_step(f"Include FL: {'yes' if include_fl else 'no'}", "info")
    print()
    _print_step("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)
            _print_step("Converting to MAT...", "done")
            result = export_to_mat(loader, output_path, include_fl=include_fl)

        _print_success(str(result))
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


def _convert_gif(args: list[str]) -> int:
    """Convert TCF to animated GIF."""
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_overlay_gif, export_to_gif

    parts = args if args else []

    if not parts:
        _print_subcommand_help(
            "GIF Animation",
            "python -m tomocube gif <file.TCF> [output.gif] [options]",
            [
                ("--overlay", "", "Create HT+FL overlay animation"),
                ("--fps", "<N>", "Frame rate (default: 10)"),
                ("--axis", "<z|y|x>", "Slice axis for animation (default: z)"),
            ],
            [
                ("Z-stack animation", "python -m tomocube gif sample.TCF"),
                ("Custom frame rate", "python -m tomocube gif sample.TCF anim.gif --fps 15"),
                ("HT+FL overlay", "python -m tomocube gif sample.TCF --overlay"),
            ],
        )
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

    mode = "HT+FL overlay" if overlay else f"{axis.upper()}-stack"
    _print_export_header("GIF Animation", tcf_path)
    _print_step(f"Mode: {mode}", "info")
    _print_step(f"Frame rate: {fps} fps", "info")
    print()
    _print_step("Loading data...")

    try:
        with TCFFileLoader(tcf_path) as loader:
            loader.load_timepoint(0)

            if overlay:
                if not loader.has_fluorescence:
                    _print_error("No fluorescence data available for overlay mode")
                    return 1
                _print_step("Generating frames...", "done")
                result = export_overlay_gif(loader, output_path, fps=fps)
            else:
                _print_step("Generating frames...", "done")
                result = export_to_gif(loader, output_path, axis=axis, fps=fps)

        _print_success(str(result))
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
