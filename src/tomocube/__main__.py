"""
Tomocube Tools CLI - Work with Tomocube TCF holotomography files.

Run 'python -m tomocube help' for usage information.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import h5py


def _get_terminal_width() -> int:
    """Get terminal width, with fallback."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


class Style:
    """Terminal styling with ANSI codes."""

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"

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

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright backgrounds
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_CYAN = "\033[106m"

    RESET = "\033[0m"

    # 256-color palette (for gradients)
    @staticmethod
    def fg256(n: int) -> str:
        """Foreground color from 256-color palette."""
        return f"\033[38;5;{n}m"

    @staticmethod
    def bg256(n: int) -> str:
        """Background color from 256-color palette."""
        return f"\033[48;5;{n}m"

    @classmethod
    def disable(cls) -> None:
        """Disable all styling."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")
        cls.fg256 = staticmethod(lambda n: "")
        cls.bg256 = staticmethod(lambda n: "")


class Icons:
    """Unicode icons and box-drawing characters."""
    # Status icons
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    ARROW_RIGHT = "▶"
    BULLET = "●"
    CIRCLE = "○"
    DIAMOND = "◆"
    DIAMOND_SM = "◇"
    STAR = "★"
    SPARK = "✦"
    LIGHTNING = "⚡"
    MICROSCOPE = "🔬"
    DNA = "🧬"
    CUBE = "📦"
    FOLDER = "📁"
    FILE = "📄"
    GEAR = "⚙"
    INFO = "ℹ"
    WARNING = "⚠"

    # Box drawing - rounded
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

    # Box drawing - heavy
    BOX_HH = "━"
    BOX_VH = "┃"

    # Double box
    DBL_H = "═"
    DBL_V = "║"
    DBL_TL = "╔"
    DBL_TR = "╗"
    DBL_BL = "╚"
    DBL_BR = "╝"

    # Arrows and pointers
    CHEVRON = "›"
    CHEVRON_DBL = "»"
    POINTER = "▸"
    POINTER_DBL = "▶"
    TRIANGLE = "▲"
    TRIANGLE_DOWN = "▼"
    TRIANGLE_RIGHT = "▷"

    # Progress/bars
    BLOCK_FULL = "█"
    BLOCK_7 = "▉"
    BLOCK_6 = "▊"
    BLOCK_5 = "▋"
    BLOCK_4 = "▌"
    BLOCK_3 = "▍"
    BLOCK_2 = "▎"
    BLOCK_1 = "▏"
    BLOCK_MED = "▓"
    BLOCK_LIGHT = "░"

    # Decorative
    DOTS = "···"
    ELLIPSIS = "…"
    WAVE = "〰"
    SPARKLE = "✨"

    @classmethod
    def disable(cls) -> None:
        """Replace unicode with ASCII fallbacks."""
        cls.CHECK = "+"
        cls.CROSS = "x"
        cls.ARROW = "->"
        cls.ARROW_RIGHT = ">"
        cls.BULLET = "*"
        cls.CIRCLE = "o"
        cls.DIAMOND = "*"
        cls.DIAMOND_SM = "o"
        cls.STAR = "*"
        cls.SPARK = "*"
        cls.LIGHTNING = "!"
        cls.MICROSCOPE = "[M]"
        cls.DNA = "[D]"
        cls.CUBE = "[#]"
        cls.FOLDER = "[/]"
        cls.FILE = "[@]"
        cls.GEAR = "[*]"
        cls.INFO = "(i)"
        cls.WARNING = "(!)"
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
        cls.BOX_HH = "="
        cls.BOX_VH = "|"
        cls.DBL_H = "="
        cls.DBL_V = "|"
        cls.DBL_TL = "+"
        cls.DBL_TR = "+"
        cls.DBL_BL = "+"
        cls.DBL_BR = "+"
        cls.CHEVRON = ">"
        cls.CHEVRON_DBL = ">>"
        cls.POINTER = ">"
        cls.POINTER_DBL = ">>"
        cls.TRIANGLE = "^"
        cls.TRIANGLE_DOWN = "v"
        cls.TRIANGLE_RIGHT = ">"
        cls.BLOCK_FULL = "#"
        cls.BLOCK_7 = "#"
        cls.BLOCK_6 = "#"
        cls.BLOCK_5 = "#"
        cls.BLOCK_4 = "#"
        cls.BLOCK_3 = "#"
        cls.BLOCK_2 = "#"
        cls.BLOCK_1 = "#"
        cls.BLOCK_MED = "#"
        cls.BLOCK_LIGHT = "."
        cls.DOTS = "..."
        cls.ELLIPSIS = "..."
        cls.WAVE = "~~~"
        cls.SPARKLE = "*"


class Spinner:
    """Animated spinner for long operations."""

    # Elegant dot spinner
    FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
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


class ProgressBar:
    """Smooth progress bar with percentage."""

    def __init__(self, total: int, width: int = 30, title: str = ""):
        self.total = total
        self.width = width
        self.title = title
        self.current = 0

    def update(self, current: int) -> None:
        self.current = current
        self._draw()

    def _draw(self) -> None:
        s = Style
        i = Icons
        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        empty = self.width - filled

        # Gradient from cyan to green
        bar = f"{s.CYAN}{i.BLOCK_FULL * filled}{s.BRIGHT_BLACK}{i.BLOCK_LIGHT * empty}{s.RESET}"
        pct_str = f"{pct * 100:5.1f}%"

        title_part = f"{self.title} " if self.title else ""
        sys.stdout.write(f"\r  {s.DIM}{title_part}{s.RESET}{bar} {s.BRIGHT_WHITE}{pct_str}{s.RESET}")
        sys.stdout.flush()

    def finish(self) -> None:
        s = Style
        i = Icons
        bar = f"{s.GREEN}{i.BLOCK_FULL * self.width}{s.RESET}"
        title_part = f"{self.title} " if self.title else ""
        sys.stdout.write(f"\r  {s.DIM}{title_part}{s.RESET}{bar} {s.GREEN}100.0%{s.RESET}\n")
        sys.stdout.flush()


# Disable colors if not a TTY or on Windows without ANSI support
if not sys.stdout.isatty():
    Style.disable()
    Icons.disable()

# Enable UTF-8 output on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )


def _styled(text: str, *styles: str) -> str:
    """Apply multiple styles to text."""
    return "".join(styles) + text + Style.RESET


def _get_content_width() -> int:
    """Get usable content width (terminal width minus margins)."""
    return min(_get_terminal_width() - 4, 80)


def _gradient_text(text: str, colors: list[int]) -> str:
    """Apply a gradient effect to text using 256-color palette."""
    s = Style
    if not colors:
        return text
    result = []
    for idx, char in enumerate(text):
        color_idx = int(idx / len(text) * len(colors))
        color_idx = min(color_idx, len(colors) - 1)
        result.append(f"{s.fg256(colors[color_idx])}{char}")
    return "".join(result) + s.RESET


def _horizontal_line(char: str = "─", width: int | None = None, color: str = "") -> str:
    """Create a horizontal line."""
    s = Style
    w = width or _get_content_width()
    line_color = color or s.BRIGHT_BLACK
    return f"  {line_color}{char * w}{s.RESET}"


def _gradient_line(width: int | None = None) -> str:
    """Create a gradient horizontal line."""
    s = Style
    w = width or _get_content_width()
    # Cyan to blue gradient
    colors = [51, 45, 39, 33, 27, 33, 39, 45, 51]
    segment = w // len(colors)
    line = ""
    for color in colors:
        line += f"{s.fg256(color)}{'━' * segment}"
    # Fill remaining
    remaining = w - (segment * len(colors))
    line += f"{s.fg256(colors[-1])}{'━' * remaining}"
    return f"  {line}{s.RESET}"


class Panel:
    """A styled panel/card component with responsive width."""

    def __init__(
        self,
        title: str = "",
        icon: str = "",
        color: str = "",
        width: int | None = None,
    ):
        self.title = title
        self.icon = icon
        self.color = color or Style.BRIGHT_CYAN
        self.width = width or _get_content_width()
        self.rows: list[tuple[str, str, str]] = []  # (label, value, suffix)

    def add_row(self, label: str, value: str, suffix: str = "") -> "Panel":
        self.rows.append((label, value, suffix))
        return self

    def add_divider(self) -> "Panel":
        self.rows.append(("__divider__", "", ""))
        return self

    def render(self) -> list[str]:
        s = Style
        i = Icons
        lines = []

        # Header with icon and title
        if self.title:
            header = f"  {self.color}{self.icon}{s.RESET} {s.BOLD}{self.title}{s.RESET}"
            lines.append("")
            lines.append(header)
            lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_H * (self.width - 2)}{s.RESET}")

        # Content rows
        for idx, (label, value, suffix) in enumerate(self.rows):
            if label == "__divider__":
                lines.append(f"  {s.BRIGHT_BLACK}{i.BOX_H * (self.width - 2)}{s.RESET}")
                continue

            is_last = idx == len(self.rows) - 1
            connector = i.BOX_BL if is_last else i.BOX_L

            # Format the row
            suffix_str = f" {s.BRIGHT_BLACK}{suffix}{s.RESET}" if suffix else ""
            lines.append(
                f"  {s.BRIGHT_BLACK}{connector}{i.BOX_H}{s.RESET} {label:12} {value}{suffix_str}"
            )

        return lines

    def print(self) -> None:
        for line in self.render():
            print(line)


def _print_logo(compact: bool = False) -> None:
    """Print the Tomocube Tools logo."""
    s = Style
    i = Icons
    width = _get_content_width()

    print()

    if not compact and width >= 60:
        # Beautiful ASCII art banner
        logo_lines = [
            "  ╭─────────────────────────────────────────────────────╮",
            "  │                                                     │",
            "  │   ████████╗ ██████╗ ███╗   ███╗ ██████╗             │",
            "  │   ╚══██╔══╝██╔═══██╗████╗ ████║██╔═══██╗            │",
            "  │      ██║   ██║   ██║██╔████╔██║██║   ██║            │",
            "  │      ██║   ██║   ██║██║╚██╔╝██║██║   ██║            │",
            "  │      ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝            │",
            "  │      ╚═╝    ╚═════╝ ╚═╝     ╚═╝ ╚═════╝             │",
            "  │                                                     │",
            "  │      ██████╗██╗   ██╗██████╗ ███████╗               │",
            "  │     ██╔════╝██║   ██║██╔══██╗██╔════╝               │",
            "  │     ██║     ██║   ██║██████╔╝█████╗                 │",
            "  │     ██║     ██║   ██║██╔══██╗██╔══╝                 │",
            "  │     ╚██████╗╚██████╔╝██████╔╝███████╗               │",
            "  │      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝               │",
            "  │                                                     │",
            "  ╰─────────────────────────────────────────────────────╯",
        ]
        # Apply cyan gradient
        for idx, line in enumerate(logo_lines):
            # Gradient from bright cyan at edges to cyan in middle
            if idx in (0, len(logo_lines) - 1):
                print(f"{s.BRIGHT_CYAN}{line}{s.RESET}")
            elif "████" in line or "═══" in line:
                print(f"{s.CYAN}{line}{s.RESET}")
            else:
                print(f"{s.BRIGHT_BLACK}{line}{s.RESET}")

        # Tagline
        print()
        tagline = "Holotomography Data Processing Tools"
        padding = (55 - len(tagline)) // 2
        print(f"  {s.BRIGHT_BLACK}│{' ' * padding}{s.RESET}{s.DIM}{tagline}{s.RESET}{s.BRIGHT_BLACK}{' ' * (55 - len(tagline) - padding - 2)}│{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}│{' ' * 20}{s.RESET}{s.BRIGHT_CYAN}v0.1.0{s.RESET}{s.BRIGHT_BLACK}{' ' * 27}│{s.RESET}")
        print()
    else:
        # Compact logo for narrow terminals
        print(_gradient_line(width))
        print()
        print(f"  {s.BRIGHT_CYAN}{s.BOLD}{i.DIAMOND} TOMOCUBE{s.RESET} {s.CYAN}TOOLS{s.RESET}  {s.BRIGHT_BLACK}v0.1.0{s.RESET}")
        print(f"  {s.DIM}Holotomography data processing{s.RESET}")
        print()
        print(_gradient_line(width))
        print()


def _print_help() -> None:
    """Print comprehensive help information."""
    s = Style
    i = Icons
    width = _get_content_width()

    _print_logo()

    # Usage section with styled box
    print(f"  {s.BRIGHT_WHITE}{s.BOLD}USAGE{s.RESET}")
    print(_gradient_line(width))
    print()
    print(f"  {s.DIM}${s.RESET} python -m tomocube {s.CYAN}<command>{s.RESET} {s.BRIGHT_BLACK}[file] [options]{s.RESET}")
    print()

    # Commands section with cards
    print(f"  {s.BRIGHT_WHITE}{s.BOLD}COMMANDS{s.RESET}")
    print(_gradient_line(width))
    print()

    # Three-column layout for commands if wide enough
    if width >= 70:
        # Each column is exactly 28 visible characters
        # Header row: ▶ + space + title, padded to 28 chars each
        print(f"  {s.BRIGHT_MAGENTA}{i.POINTER_DBL}{s.RESET} {s.BOLD}Visualization{s.RESET}            {s.BRIGHT_BLUE}{i.POINTER_DBL}{s.RESET} {s.BOLD}Information{s.RESET}             {s.BRIGHT_GREEN}{i.POINTER_DBL}{s.RESET} {s.BOLD}Export{s.RESET}")
        # Vertical bar row: │ padded to 28 chars each
        print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}                            {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}                            {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
        # Content rows
        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}view{s.RESET}   {s.DIM}slice viewer{s.RESET}       {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}info{s.RESET}  {s.DIM}metadata{s.RESET}           {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}tiff{s.RESET}  {s.DIM}TIFF stack{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}view3d{s.RESET} {s.DIM}volume render{s.RESET}      {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}help{s.RESET}  {s.DIM}this help{s.RESET}          {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} {s.CYAN}mat{s.RESET}   {s.DIM}MATLAB .mat{s.RESET}")
        print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}slice{s.RESET}  {s.DIM}HT/FL compare{s.RESET}                                   {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} {s.CYAN}gif{s.RESET}   {s.DIM}animation{s.RESET}")
    else:
        # Single column for narrow terminals
        print(f"  {s.BRIGHT_MAGENTA}{i.POINTER}{s.RESET} {s.BOLD}Visualization{s.RESET}")
        print(f"    {s.CYAN}view{s.RESET}    {s.DIM}Interactive orthogonal slice viewer{s.RESET}")
        print(f"    {s.CYAN}view3d{s.RESET}  {s.DIM}3D volume rendering (napari/pyvista){s.RESET}")
        print(f"    {s.CYAN}slice{s.RESET}   {s.DIM}Side-by-side HT/FL comparison{s.RESET}")
        print()
        print(f"  {s.BRIGHT_BLUE}{i.POINTER}{s.RESET} {s.BOLD}Information{s.RESET}")
        print(f"    {s.CYAN}info{s.RESET}    {s.DIM}Display file metadata & structure{s.RESET}")
        print(f"    {s.CYAN}help{s.RESET}    {s.DIM}Show this help message{s.RESET}")
        print()
        print(f"  {s.BRIGHT_GREEN}{i.POINTER}{s.RESET} {s.BOLD}Export{s.RESET}")
        print(f"    {s.CYAN}tiff{s.RESET}    {s.DIM}Export to multi-page TIFF stack{s.RESET}")
        print(f"    {s.CYAN}mat{s.RESET}     {s.DIM}Export to MATLAB .mat format{s.RESET}")
        print(f"    {s.CYAN}gif{s.RESET}     {s.DIM}Create animated GIF{s.RESET}")
    print()

    # Options section with styled headers
    print(f"  {s.BRIGHT_WHITE}{s.BOLD}OPTIONS{s.RESET}")
    print(_gradient_line(width))
    print()

    # TIFF options in a nice card (50 visible chars between │ and │)
    print(f"  {s.YELLOW}{i.SPARK}{s.RESET} {s.BOLD}tiff{s.RESET} {s.BRIGHT_BLACK}─ Export to TIFF stack{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * 50}{i.BOX_TR}{s.RESET}")
    # 50 chars:   --fl <CH>       Export fluorescence channel     
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--fl{s.RESET} {s.DIM}<CH>{s.RESET}       Export fluorescence channel     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    # 50 chars:   --32bit         32-bit float (default)          
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--32bit{s.RESET}         32-bit float {s.BRIGHT_BLACK}(default){s.RESET}          {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    # 50 chars:   --16bit         16-bit (requires --normalize)   
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--16bit{s.RESET}         16-bit {s.BRIGHT_BLACK}(requires --normalize){s.RESET}   {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    # 50 chars:   --normalize     Normalize for visualization     
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--normalize{s.RESET}     Normalize for visualization     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H * 50}{i.BOX_BR}{s.RESET}")
    print()

    # MAT options
    print(f"  {s.YELLOW}{i.SPARK}{s.RESET} {s.BOLD}mat{s.RESET} {s.BRIGHT_BLACK}─ Export to MATLAB format{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * 50}{i.BOX_TR}{s.RESET}")
    # 50 chars:   --no-fl         Exclude fluorescence data       
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--no-fl{s.RESET}         Exclude fluorescence data       {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H * 50}{i.BOX_BR}{s.RESET}")
    print()

    # GIF options
    print(f"  {s.YELLOW}{i.SPARK}{s.RESET} {s.BOLD}gif{s.RESET} {s.BRIGHT_BLACK}─ Create animated GIF{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * 50}{i.BOX_TR}{s.RESET}")
    # 50 chars:   --overlay       HT+FL overlay animation         
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--overlay{s.RESET}       HT+FL overlay animation         {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    # 50 chars:   --fps <N>       Frame rate (default: 10)        
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--fps{s.RESET} {s.DIM}<N>{s.RESET}       Frame rate {s.BRIGHT_BLACK}(default: 10){s.RESET}        {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    # 50 chars:   --axis <z|y|x>  Slice axis (default: z)         
    print(f"    {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}  {s.CYAN}--axis{s.RESET} {s.DIM}<z|y|x>{s.RESET}  Slice axis {s.BRIGHT_BLACK}(default: z){s.RESET}         {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"    {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H * 50}{i.BOX_BR}{s.RESET}")
    print()

    # Keyboard shortcuts in a styled grid
    print(f"  {s.BRIGHT_WHITE}{s.BOLD}VIEWER SHORTCUTS{s.RESET}")
    print(_gradient_line(width))
    print()

    _print_shortcut_grid([
        [("↑↓", "Navigate Z"), ("A", "Auto contrast"), ("D", "Distance"), ("S", "Save PNG")],
        [("⎚", "Scroll Z"), ("G", "Global contrast"), ("P", "Polygon"), ("M", "Save MIP")],
        [("Click", "Select"), ("I", "Invert cmap"), ("C", "Clear"), ("F", "Toggle FL")],
        [("1-6", "Colormap"), ("R", "Reset view"), ("Q", "Quit"), ("", "")],
    ])
    print()

    # Examples in a nice styled section
    print(f"  {s.BRIGHT_WHITE}{s.BOLD}EXAMPLES{s.RESET}")
    print(_gradient_line(width))
    print()

    examples = [
        ("View a TCF file", "python -m tomocube view sample.TCF"),
        ("Export to 32-bit TIFF", "python -m tomocube tiff sample.TCF --32bit"),
        ("Create HT+FL overlay animation", "python -m tomocube gif sample.TCF --overlay --fps 15"),
    ]

    for comment, cmd in examples:
        print(f"  {s.BRIGHT_BLACK}# {comment}{s.RESET}")
        print(f"  {s.DIM}${s.RESET} {s.CYAN}{cmd}{s.RESET}")
        print()

    # Footer
    print(_gradient_line(width))
    print(f"  {s.DIM}Documentation: {s.RESET}{s.BRIGHT_BLACK}https://github.com/tomocube/tomocube-tools{s.RESET}")
    print()


def _print_shortcut_grid(rows: list[list[tuple[str, str]]]) -> None:
    """Print keyboard shortcuts in a grid."""
    s = Style
    i = Icons
    for row in rows:
        parts = []
        for key, desc in row:
            if key:
                parts.append(f"{s.CYAN}{key:7}{s.RESET} {s.DIM}{desc:15}{s.RESET}")
            else:
                parts.append(" " * 23)
        print("  " + "".join(parts))


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
    width = _get_content_width()

    _print_logo(compact=True)

    print(f"  {s.BRIGHT_WHITE}{s.BOLD}QUICK START{s.RESET}")
    print(_gradient_line(width))
    print()
    print(f"  {s.DIM}${s.RESET} python -m tomocube {s.CYAN}<command>{s.RESET} {s.BRIGHT_BLACK}[file] [options]{s.RESET}")
    print()

    # Commands in a nice grid
    print(f"  {s.BRIGHT_BLACK}{i.BOX_TL}{i.BOX_H * 26}{i.BOX_T}{i.BOX_H * 26}{i.BOX_TR}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}view{s.RESET}   {s.DIM}slice viewer{s.RESET}      {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}tiff{s.RESET}   {s.DIM}TIFF export{s.RESET}       {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}view3d{s.RESET} {s.DIM}volume render{s.RESET}     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}mat{s.RESET}    {s.DIM}MATLAB export{s.RESET}     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}slice{s.RESET}  {s.DIM}HT/FL compare{s.RESET}     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}gif{s.RESET}    {s.DIM}animation{s.RESET}         {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET} {s.CYAN}info{s.RESET}   {s.DIM}file metadata{s.RESET}     {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}                          {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H * 26}{i.BOX_B}{i.BOX_H * 26}{i.BOX_BR}{s.RESET}")
    print()

    print(f"  {s.BRIGHT_BLACK}{i.INFO}{s.RESET} Run {s.CYAN}python -m tomocube help{s.RESET} for detailed options")
    print()


def _print_info(file_path: str) -> int:
    """Print formatted file information."""
    from tomocube.core import TCFFile
    from tomocube.processing import discover_related_metadata

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

    # Discover related metadata files
    related_meta = discover_related_metadata(tcf_path)

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

    # Related metadata sections (from external files)
    if related_meta:
        # Experiment metadata
        if "experiment" in related_meta:
            exp = related_meta["experiment"]
            print(f"  {s.BRIGHT_BLUE}{i.POINTER}{s.RESET} {s.BOLD}Experiment{s.RESET}  {s.BRIGHT_BLACK}(.experiment){s.RESET}")
            print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
            if "experimentTitle" in exp:
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Title        {s.CYAN}{exp['experimentTitle']}{s.RESET}")
            if "createdDate" in exp:
                # Format date from YYYYMMDD to YYYY-MM-DD
                date_str = str(exp["createdDate"])
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Date         {date_str}")
            if "user" in exp and exp["user"]:
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} User         {exp['user']}")
            if "medium" in exp:
                medium = exp["medium"]
                medium_name = medium.get("mediumName", "?")
                medium_ri = medium.get("mediumRI", "?")
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Medium       {medium_name} {s.BRIGHT_BLACK}(RI: {medium_ri}){s.RESET}")
            if "sampleType" in exp:
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Sample Type  {exp['sampleType']}")
            if "vesselModel" in exp:
                print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Vessel       {exp['vesselModel']}")
            if "hTLightChannel" in exp:
                print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} HT Light Ch  {exp['hTLightChannel']}")
            print()

        # Vessel metadata
        if "vessel" in related_meta:
            vessel = related_meta["vessel"]
            print(f"  {s.BRIGHT_YELLOW}{i.POINTER}{s.RESET} {s.BOLD}Vessel{s.RESET}  {s.BRIGHT_BLACK}(.vessel){s.RESET}")
            print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
            if "vessel" in vessel:
                v = vessel["vessel"]
                if "name" in v:
                    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Name         {s.CYAN}{v['name']}{s.RESET}")
                elif "model" in v:
                    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Model        {s.CYAN}{v['model']}{s.RESET}")
                if "size" in v:
                    size = v["size"]
                    # Size can be [width, height] list or dict
                    if isinstance(size, list) and len(size) >= 2:
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Size         {size[0]} × {size[1]} {s.BRIGHT_BLACK}mm{s.RESET}")
                    elif isinstance(size, dict):
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Size         {size.get('width', '?')} × {size.get('height', '?')} {s.BRIGHT_BLACK}mm{s.RESET}")
                if "NA" in v:
                    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} NA           {v['NA']}")
                if "AFOffset" in v:
                    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} AF Offset    {v['AFOffset']} {s.BRIGHT_BLACK}μm{s.RESET}")
            if "well" in vessel:
                well = vessel["well"]
                if "rows" in well and "columns" in well:
                    print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Wells        {well['rows']} × {well['columns']} {s.BRIGHT_BLACK}grid{s.RESET}")
                if "size" in well:
                    well_size = well["size"]
                    if isinstance(well_size, list) and len(well_size) >= 2:
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Well Size    {well_size[0]} × {well_size[1]} {s.BRIGHT_BLACK}mm{s.RESET}")
                if "spacing" in well:
                    spacing = well["spacing"]
                    if isinstance(spacing, list) and len(spacing) >= 2:
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Well Spacing {spacing[0]} × {spacing[1]} {s.BRIGHT_BLACK}mm{s.RESET}")
            print()

        # Profile metadata
        if "profiles" in related_meta:
            profiles = related_meta["profiles"]
            profile_names = list(profiles.keys())
            print(f"  {s.BRIGHT_MAGENTA}{i.POINTER}{s.RESET} {s.BOLD}Profiles{s.RESET}  {s.BRIGHT_BLACK}(profile/*.prf){s.RESET}")
            print(f"  {s.BRIGHT_BLACK}{i.BOX_V}{s.RESET}")
            print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Available    {s.CYAN}{', '.join(profile_names)}{s.RESET}")
            # Show key settings from img profile if available
            if "img" in profiles:
                img = profiles["img"]
                if "DefaultParameters" in img:
                    defaults = img["DefaultParameters"]
                    if "DefaultStep" in defaults:
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Z Step       {defaults['DefaultStep']} {s.BRIGHT_BLACK}μm{s.RESET}")
                    if "DefaultSlices" in defaults:
                        print(f"  {s.BRIGHT_BLACK}{i.BOX_L}{i.BOX_H}{s.RESET} Z Slices     {defaults['DefaultSlices']}")
                if "SupportedNA" in img:
                    na_list = img["SupportedNA"]
                    if isinstance(na_list, dict):
                        na_values = [str(v) for v in na_list.values() if v]
                        if na_values:
                            print(f"  {s.BRIGHT_BLACK}{i.BOX_BL}{i.BOX_H}{s.RESET} Supported NA {', '.join(na_values)}")
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

    elif command == "view3d":
        return _view_3d(args_list)

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
                    result = export_overlay_gif(loader, output_path, axis=axis, fps=fps)
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


def _view_3d(args: list[str]) -> int:
    """Launch 3D volume viewer."""
    s = Style
    i = Icons

    if not args:
        _print_error("Missing file path")
        print(f"  Usage: python -m tomocube view3d <file.TCF> [--backend napari|pyvista]")
        print()
        print(f"  {s.BRIGHT_BLACK}Options:{s.RESET}")
        print(f"    {s.CYAN}--backend{s.RESET}    Viewer backend: {s.DIM}napari{s.RESET} (default) or {s.DIM}pyvista{s.RESET}")
        print()
        print(f"  {s.BRIGHT_BLACK}Install 3D dependencies:{s.RESET}")
        print(f"    {s.DIM}pip install 'tomocube-tools[3d]'{s.RESET}")
        print()
        return 1

    parts = args if args else []
    tcf_path = parts[0]
    backend = "napari"

    idx = 1
    while idx < len(parts):
        if parts[idx] == "--backend" and idx + 1 < len(parts):
            backend = parts[idx + 1].lower()
            idx += 2
        elif parts[idx] == "--pyvista":
            backend = "pyvista"
            idx += 1
        elif parts[idx] == "--napari":
            backend = "napari"
            idx += 1
        else:
            idx += 1

    if not Path(tcf_path).exists():
        _print_error(f"File not found: {tcf_path}")
        return 1

    print()
    print(f"  {s.BRIGHT_CYAN}{Icons.CUBE}{s.RESET} {s.BOLD}3D Volume Viewer{s.RESET}")
    print(f"  {s.BRIGHT_BLACK}{'─' * 50}{s.RESET}")
    print(f"  {s.DIM}File:{s.RESET}    {Path(tcf_path).name}")
    print(f"  {s.DIM}Backend:{s.RESET} {backend}")
    print()

    try:
        from tomocube.viewer.viewer_3d import view_3d
        view_3d(tcf_path, backend=backend)
        return 0
    except ImportError as e:
        _print_error(str(e))
        print()
        print(f"  {s.BRIGHT_BLACK}To install 3D viewer dependencies:{s.RESET}")
        print(f"    {s.CYAN}pip install 'tomocube-tools[3d]'{s.RESET}")
        print()
        return 1
    except Exception as e:
        _print_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
