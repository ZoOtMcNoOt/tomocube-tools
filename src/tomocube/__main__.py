"""
Tomocube Tools CLI - Work with Tomocube TCF holotomography files.

Run 'python -m tomocube help' for usage information.
"""

from __future__ import annotations

import sys
import time
import threading
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


class Icons:
    """Unicode icons and box-drawing characters."""
    # Status icons
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    BULLET = "●"
    CIRCLE = "○"
    DIAMOND = "◆"
    STAR = "★"
    SPARK = "✦"
    
    # Box drawing
    BOX_H = "─"
    BOX_V = "│"
    BOX_TL = "╭"
    BOX_TR = "╮"
    BOX_BL = "╰"
    BOX_BR = "╯"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"
    
    # Double box
    DBL_H = "═"
    DBL_V = "║"
    
    # Arrows and pointers
    CHEVRON = "›"
    POINTER = "▸"
    TRIANGLE = "▲"
    
    # Progress
    BLOCK_FULL = "█"
    BLOCK_MED = "▓"
    BLOCK_LIGHT = "░"
    
    # Decorative
    DOTS = "···"
    ELLIPSIS = "…"
    
    @classmethod
    def disable(cls) -> None:
        """Replace unicode with ASCII fallbacks."""
        cls.CHECK = "+"
        cls.CROSS = "x"
        cls.ARROW = "->"
        cls.BULLET = "*"
        cls.CIRCLE = "o"
        cls.DIAMOND = "*"
        cls.STAR = "*"
        cls.SPARK = "*"
        cls.BOX_H = "-"
        cls.BOX_V = "|"
        cls.BOX_TL = "+"
        cls.BOX_TR = "+"
        cls.BOX_BL = "+"
        cls.BOX_BR = "+"
        cls.BOX_T = "+"
        cls.BOX_B = "+"
        cls.BOX_L = "+"
        cls.BOX_R = "+"
        cls.BOX_X = "+"
        cls.DBL_H = "="
        cls.DBL_V = "|"
        cls.CHEVRON = ">"
        cls.POINTER = ">"
        cls.TRIANGLE = "^"
        cls.BLOCK_FULL = "#"
        cls.BLOCK_MED = "#"
        cls.BLOCK_LIGHT = "."
        cls.DOTS = "..."
        cls.ELLIPSIS = "..."


class Spinner:
    """Animated spinner for long operations."""
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FALLBACK = ["|", "/", "-", "\\"]
    
    def __init__(self, message: str = ""):
        self.message = message
        self._stop = False
        self._thread: threading.Thread | None = None
        self._frames = self.FRAMES if sys.stdout.isatty() else self.FALLBACK
    
    def __enter__(self) -> "Spinner":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
    
    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def stop(self, success: bool = True) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.5)
        # Clear spinner line and print final status
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
    
    def _spin(self) -> None:
        s = Style
        i = 0
        while not self._stop:
            frame = self._frames[i % len(self._frames)]
            sys.stdout.write(f"\r  {s.CYAN}{frame}{s.RESET} {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1


# Disable colors if not a TTY or on Windows without ANSI support
if not sys.stdout.isatty():
    Style.disable()
    Icons.disable()


def _styled(text: str, *styles: str) -> str:
    """Apply multiple styles to text."""
    return "".join(styles) + text + Style.RESET


def _box(content: list[str], width: int = 54, title: str = "") -> list[str]:
    """Create a box around content."""
    i = Icons
    s = Style
    lines = []
    
    # Top border with optional title
    if title:
        title_str = f" {title} "
        pad_left = (width - len(title_str) - 2) // 2
        pad_right = width - len(title_str) - 2 - pad_left
        lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * pad_left}{s.RESET}{s.BOLD}{title_str}{s.RESET}{s.BRIGHT_BLACK}{i.BOX_H * pad_right}{i.BOX_TR}{s.RESET}")
    else:
        lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * width}{i.BOX_TR}{s.RESET}")
    
    # Content
    for line in content:
        # Strip ANSI codes for length calculation
        import re
        clean_line = re.sub(r'\033\[[0-9;]*m', '', line)
        padding = width - len(clean_line) - 2
        lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {line}{' ' * padding}{s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    
    # Bottom border
    lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H * width}{i.BOX_BR}{s.RESET}")
    
    return lines


def _print_logo() -> None:
    """Print the Tomocube Tools logo."""
    s = Style
    i = Icons
    
    print()
    # Stylized header
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{i.DIAMOND} TOMOCUBE{s.RESET} {s.DIM}Tools{s.RESET} {s.BRIGHT_BLACK}v0.1.0{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}Holotomography data processing{s.RESET}")
    print()


def _print_help() -> None:
    """Print comprehensive help information."""
    s = Style
    i = Icons

    _print_logo()
    
    # Usage box
    print(f"  {s.BOLD}USAGE{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print(f"  python -m tomocube {s.CYAN}<command>{s.RESET} {s.DIM}[file] [options]{s.RESET}")
    print()

    # Commands section with grouped layout
    print(f"  {s.BOLD}COMMANDS{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    
    # Visualization group
    print(f"  {s.BRIGHT_MAGENTA}{i.POINTER}{s.RESET} {s.BOLD}Visualization{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}view{s.RESET}  {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Interactive 3D orthogonal slice viewer")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}slice{s.RESET} {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Side-by-side HT/FL comparison")
    print()
    
    # Information group
    print(f"  {s.BRIGHT_BLUE}{i.POINTER}{s.RESET} {s.BOLD}Information{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}info{s.RESET}  {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Display file metadata & structure")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}help{s.RESET}          {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Show this help message")
    print()
    
    # Export group
    print(f"  {s.BRIGHT_GREEN}{i.POINTER}{s.RESET} {s.BOLD}Export{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}tiff{s.RESET}  {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Export to multi-page TIFF stack")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}mat{s.RESET}   {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Export to MATLAB .mat format")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}gif{s.RESET}   {s.DIM}<file>{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET}  Create animated GIF")
    print()

    # Options section
    print(f"  {s.BOLD}OPTIONS{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    
    # TIFF options
    print(f"  {s.YELLOW}{i.BULLET}{s.RESET} {s.BOLD}tiff{s.RESET}")
    print(f"      {s.CYAN}--fl{s.RESET} {s.DIM}<CH>{s.RESET}      Export fluorescence channel (e.g., CH0)")
    print(f"      {s.CYAN}--32bit{s.RESET}        32-bit float, physical RI values {s.BRIGHT_BLACK}(default){s.RESET}")
    print(f"      {s.CYAN}--16bit{s.RESET}        16-bit output {s.BRIGHT_BLACK}(requires --normalize){s.RESET}")
    print(f"      {s.CYAN}--normalize{s.RESET}    Normalize values for visualization")
    print()
    
    # MAT options
    print(f"  {s.YELLOW}{i.BULLET}{s.RESET} {s.BOLD}mat{s.RESET}")
    print(f"      {s.CYAN}--no-fl{s.RESET}        Exclude fluorescence data")
    print()
    
    # GIF options
    print(f"  {s.YELLOW}{i.BULLET}{s.RESET} {s.BOLD}gif{s.RESET}")
    print(f"      {s.CYAN}--overlay{s.RESET}      HT+FL overlay animation")
    print(f"      {s.CYAN}--fps{s.RESET} {s.DIM}<N>{s.RESET}      Frame rate {s.BRIGHT_BLACK}(default: 10){s.RESET}")
    print(f"      {s.CYAN}--axis{s.RESET} {s.DIM}<z|y|x>{s.RESET} Slice axis {s.BRIGHT_BLACK}(default: z){s.RESET}")
    print()

    # Keyboard shortcuts in a nice grid
    print(f"  {s.BOLD}VIEWER SHORTCUTS{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    
    _print_shortcut_grid([
        [("↑↓", "Navigate Z"), ("A", "Auto contrast"), ("D", "Distance"), ("S", "Save PNG")],
        [("⎚", "Scroll Z"), ("G", "Global contrast"), ("P", "Polygon"), ("M", "Save MIP")],
        [("Click", "Select"), ("I", "Invert cmap"), ("C", "Clear"), ("F", "Toggle FL")],
        [("1-6", "Colormap"), ("R", "Reset view"), ("Q", "Quit"), ("", "")],
    ])
    print()

    # Examples
    print(f"  {s.BOLD}EXAMPLES{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    print(f"  {s.BRIGHT_BLACK}# View a TCF file{s.RESET}")
    print(f"  {s.DIM}${s.RESET} python -m tomocube view sample.TCF")
    print()
    print(f"  {s.BRIGHT_BLACK}# Export to 32-bit TIFF{s.RESET}")
    print(f"  {s.DIM}${s.RESET} python -m tomocube tiff sample.TCF --32bit")
    print()
    print(f"  {s.BRIGHT_BLACK}# Create HT+FL overlay animation{s.RESET}")
    print(f"  {s.DIM}${s.RESET} python -m tomocube gif sample.TCF --overlay --fps 15")
    print()


def _print_shortcut_grid(rows: list[list[tuple[str, str]]]) -> None:
    """Print keyboard shortcuts in a grid."""
    s = Style
    i = Icons
    for row in rows:
        parts = []
        for key, desc in row:
            if key:
                parts.append(f"  {s.CYAN}{key:6}{s.RESET} {s.DIM}{desc:12}{s.RESET}")
            else:
                parts.append(" " * 20)
        print("".join(parts))


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
    i = Icons
    _print_logo()

    print(f"  {s.BOLD}USAGE{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print(f"  python -m tomocube {s.CYAN}<command>{s.RESET} {s.DIM}[file] [options]{s.RESET}")
    print()
    
    print(f"  {s.BOLD}COMMANDS{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    print(f"    {s.CYAN}view{s.RESET}   Interactive 3D viewer     {s.CYAN}tiff{s.RESET}   Export TIFF stack")
    print(f"    {s.CYAN}slice{s.RESET}  HT/FL comparison          {s.CYAN}mat{s.RESET}    Export MATLAB .mat")
    print(f"    {s.CYAN}info{s.RESET}   File metadata             {s.CYAN}gif{s.RESET}    Create animation")
    print()
    print(f"  {s.DIM}Run{s.RESET} python -m tomocube help {s.DIM}for detailed options{s.RESET}")
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

    i = Icons
    
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{i.DIAMOND} TCF File Info{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()

    # File section
    print(f"  {s.BRIGHT_BLUE}{i.POINTER}{s.RESET} {s.BOLD}File{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Name       {s.GREEN}{tcf_path.name}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Location   {s.BRIGHT_BLACK}{tcf_path.parent}{s.RESET}")
    print()

    # Holotomography section
    print(f"  {s.BRIGHT_MAGENTA}{i.POINTER}{s.RESET} {s.BOLD}Holotomography{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Volume     {s.YELLOW}{ht_z}{s.RESET} × {s.YELLOW}{ht_y}{s.RESET} × {s.YELLOW}{ht_x}{s.RESET}  {s.BRIGHT_BLACK}(Z × Y × X){s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Resolution {res_x:.3f} × {res_y:.3f} × {res_z:.3f} {s.BRIGHT_BLACK}μm/px{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} FOV        {fov_x:.1f} × {fov_y:.1f} × {fov_z:.1f} {s.BRIGHT_BLACK}μm{s.RESET}")
    if info.ri_min is not None and info.ri_max is not None:
        print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} RI Range   {s.CYAN}{info.ri_min:.4f}{s.RESET} {i.ARROW} {s.CYAN}{info.ri_max:.4f}{s.RESET}")
    else:
        # Close the box
        pass
    print()

    # Optics section
    print(f"  {s.BRIGHT_YELLOW}{i.POINTER}{s.RESET} {s.BOLD}Optics{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Magnification  {s.YELLOW}{info.magnification or '?'}×{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} NA             {info.numerical_aperture or '?'}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Medium RI      {info.medium_ri or '?'}")
    print()

    # Acquisition section
    print(f"  {s.BRIGHT_CYAN}{i.POINTER}{s.RESET} {s.BOLD}Acquisition{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Timepoints   {len(info.timepoints)}")
    print()

    # Fluorescence section
    if info.has_fluorescence:
        print(f"  {s.BRIGHT_GREEN}{i.POINTER}{s.RESET} {s.BOLD}Fluorescence{s.RESET}  {s.GREEN}{i.CHECK} Available{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Channels     {s.CYAN}{', '.join(info.fl_channels)}{s.RESET}")
        for idx, (ch, shape) in enumerate(info.fl_shapes.items()):
            fl_z, fl_y, fl_x = shape
            prefix = i.BOX_L if idx < len(info.fl_shapes) - 1 else i.BOX_L
            print(f"  {s.BRIGHT_BLACK}{prefix}{i.BOX_H}{s.RESET} {ch} Volume   {fl_z} × {fl_y} × {fl_x}")
        fl_res = info.fl_resolution
        print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Resolution   {fl_res[2]:.3f} × {fl_res[1]:.3f} × {fl_res[0]:.3f} {s.BRIGHT_BLACK}μm/px{s.RESET}")
    else:
        print(f"  {s.BRIGHT_BLACK}{i.POINTER}{s.RESET} {s.BOLD}Fluorescence{s.RESET}  {s.BRIGHT_BLACK}{i.CROSS} Not available{s.RESET}")

    print()
    return 0


def _print_error(message: str) -> None:
    """Print an error message."""
    s = Style
    i = Icons
    print()
    print(f"  {s.RED}{i.CROSS}{s.RESET} {s.RED}{s.BOLD}Error{s.RESET} {message}")
    print()


def _print_success(path: str) -> None:
    """Print a success message."""
    s = Style
    i = Icons
    print(f"  {s.GREEN}{i.CHECK}{s.RESET} {s.GREEN}{s.BOLD}Saved{s.RESET} {path}")
    print()


def _print_step(message: str, status: str = "working") -> None:
    """Print a step in a process."""
    s = Style
    i = Icons
    if status == "working":
        print(f"  {s.YELLOW}{i.CHEVRON}{s.RESET} {message}")
    elif status == "done":
        print(f"  {s.GREEN}{i.CHECK}{s.RESET} {message}")
    elif status == "info":
        print(f"  {s.BRIGHT_BLACK}{i.CIRCLE}{s.RESET} {s.DIM}{message}{s.RESET}")


def _print_export_header(title: str, input_file: str) -> None:
    """Print export operation header."""
    s = Style
    i = Icons
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{i.DIAMOND} {title}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
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
    i = Icons
    
    print()
    print(f"  {s.BRIGHT_CYAN}{s.BOLD}{i.DIAMOND} {name}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print()
    
    print(f"  {s.BOLD}USAGE{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
    print(f"  {s.DIM}${s.RESET} {usage}")
    print()

    if options:
        print(f"  {s.BOLD}OPTIONS{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
        for opt, arg, desc in options:
            if arg:
                print(f"    {s.CYAN}{opt}{s.RESET} {s.DIM}{arg}{s.RESET}")
                print(f"        {s.BRIGHT_BLACK}{i.ARROW}{s.RESET} {desc}")
            else:
                print(f"    {s.CYAN}{opt}{s.RESET}  {s.BRIGHT_BLACK}{i.ARROW}{s.RESET} {desc}")
        print()

    if examples:
        print(f"  {s.BOLD}EXAMPLES{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_H * 54}{s.RESET}")
        for comment, cmd in examples:
            print(f"  {s.BRIGHT_BLACK}# {comment}{s.RESET}")
            print(f"  {s.DIM}${s.RESET} {cmd}")
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
                ("--16bit", "", "16-bit output (requires --normalize)"),
                ("--32bit", "", "32-bit float output (default, preserves RI values)"),
                ("--normalize", "", "Normalize values for visualization (required for 16-bit)"),
            ],
            [
                ("Export with physical RI values", "python -m tomocube tiff sample.TCF"),
                ("Export normalized for visualization", "python -m tomocube tiff sample.TCF output.tiff --normalize"),
                ("Export 16-bit normalized", "python -m tomocube tiff sample.TCF output.tiff --16bit --normalize"),
                ("Export FL channel", "python -m tomocube tiff sample.TCF fl.tiff --fl CH0"),
            ],
        )
        return 1

    tcf_path = parts[0]
    output_path = None
    channel = "ht"
    bit_depth = 32  # Default to 32-bit to preserve physical values
    normalize = False  # Default to preserving physical RI values

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
        elif parts[i] == "--normalize":
            normalize = True
            i += 1
        elif not output_path:
            output_path = parts[i]
            i += 1
        else:
            i += 1

    # 16-bit requires normalization
    if bit_depth == 16 and not normalize:
        _print_error("--16bit requires --normalize (cannot store RI values 1.33-1.40 in 16-bit without normalization)")
        return 1

    if not output_path:
        output_path = Path(tcf_path).stem + f"_{channel}.tiff"

    if not Path(tcf_path).exists():
        _print_error(f"File not found: {tcf_path}")
        return 1

    _print_export_header("TIFF Export", tcf_path)
    _print_step(f"Channel: {channel}", "info")
    _print_step(f"Bit depth: {bit_depth}", "info")
    if normalize:
        _print_step("Mode: Normalized (0-1 or 0-65535)", "info")
    else:
        _print_step("Mode: Physical RI values preserved", "info")
    print()

    try:
        with TCFFileLoader(tcf_path) as loader:
            with Spinner("Loading data..."):
                loader.load_timepoint(0)
            _print_step("Data loaded", "done")

            with Spinner("Converting to TIFF..."):
                result = export_to_tiff(loader, output_path, channel=channel, bit_depth=bit_depth, normalize=normalize)
            _print_step("Conversion complete", "done")

        print()
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

    try:
        with TCFFileLoader(tcf_path) as loader:
            with Spinner("Loading data..."):
                loader.load_timepoint(0)
            _print_step("Data loaded", "done")
            
            with Spinner("Converting to MAT..."):
                result = export_to_mat(loader, output_path, include_fl=include_fl)
            _print_step("Conversion complete", "done")
        
        print()
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

    try:
        with TCFFileLoader(tcf_path) as loader:
            with Spinner("Loading data..."):
                loader.load_timepoint(0)
            _print_step("Data loaded", "done")

            if overlay:
                if not loader.has_fluorescence:
                    _print_error("No fluorescence data available for overlay mode")
                    return 1
                with Spinner("Generating frames..."):
                    result = export_overlay_gif(loader, output_path, fps=fps)
            else:
                with Spinner("Generating frames..."):
                    result = export_to_gif(loader, output_path, axis=axis, fps=fps)
        
        _print_step("Animation complete", "done")
        print()
        _print_success(str(result))
        return 0
    except Exception as e:
        _print_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
