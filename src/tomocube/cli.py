"""Command-line inspection, viewers, calibrated analysis, and exports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import sys


class _CommandParser(argparse.ArgumentParser):
    """Allow options between input/output paths on every supported Python.

    argparse cannot intermix arguments at a tree's root when it has subparsers.
    Enable its standard intermixed parser at the leaves instead. The guard is
    needed because that implementation calls ``parse_known_args`` itself.
    """

    _intermixing = False

    def parse_known_args(self, args=None, namespace=None):
        if self._intermixing:
            return super().parse_known_args(args, namespace)
        self._intermixing = True
        try:
            return self.parse_known_intermixed_args(args, namespace)
        finally:
            self._intermixing = False


def _nonnegative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _positive_integer(value: str) -> int:
    number = _nonnegative_integer(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _frame_rate(value: str) -> int:
    number = _positive_integer(value)
    if number > 100:
        raise argparse.ArgumentTypeError("fps must be between 1 and 100")
    return number


def _finite_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number")
    return number


def _filename_prefix(value: str) -> str:
    if not value or value in (".", "..") or any(char in value for char in '/\\\0:'):
        raise argparse.ArgumentTypeError("prefix must be a filename, without path separators")
    return value


def _channel_name(value: str) -> str:
    if not value.strip() or any(char in value for char in '/\\\0:'):
        raise argparse.ArgumentTypeError("channel must be a non-empty name without path separators")
    return value


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", metavar="file.TCF", help="input TCF acquisition")
    parser.add_argument("--timepoint", type=_nonnegative_integer, default=0, metavar="N",
                        help="zero-based acquisition index in numeric key order (default: 0)")


def _add_alignment(parser: argparse.ArgumentParser, *, default: str | None = "start") -> None:
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--z-offset-mode", choices=("start", "center", "auto"), default=default,
                        help="metadata placement or intensity Z centering; auto is not image matching")
    alignment.add_argument("--registration", metavar="JSON", help="apply a saved image alignment for this acquisition and channel")


def build_parser() -> argparse.ArgumentParser:
    """Build the single command tree; no data or optional GUI imports occur here."""
    from tomocube import __version__

    parser = argparse.ArgumentParser(
        prog="tomocube", allow_abbrev=False,
        description=f"Tomocube Tools v{__version__}: inspect, view, measure, and export calibrated TCF data.",
        epilog="Select an acquisition with --timepoint N. Example: tomocube gif --fl CH1 sample.TCF.",
    )
    parser.add_argument("-V", "--verbose", action="store_true", help="send diagnostic details to stderr")
    parser.add_argument("--version", "-v", action="version", version=f"tomocube-tools v{__version__}")
    commands = parser.add_subparsers(dest="command", parser_class=_CommandParser)

    def command(name: str, help_text: str, handler):
        child = commands.add_parser(name, help=help_text, description=help_text, allow_abbrev=False)
        child.add_argument("-V", "--verbose", action="store_true", default=argparse.SUPPRESS,
                           help="send diagnostic details to stderr")
        child.set_defaults(handler=handler)
        return child

    info = command("info", "Inspect acquisition headers and related metadata without loading volumes.", _info)
    info.add_argument("file", metavar="file.TCF")
    info.add_argument("--json", action="store_true", help="write the complete inventory as JSON")

    for name, description in (("view", "Open the orthogonal 2D viewer."),
                              ("slice", "Compare HT, fluorescence, and overlay slices."),
                              ("view3d", "Open the napari 3D viewer (requires the 3d extra).")):
        viewer = command(name, description, _view3d if name == "view3d" else _view)
        _add_selection(viewer)
        viewer.add_argument("--fl", type=_channel_name, metavar="CHANNEL", help="fluorescence channel (default: first available)")
        _add_alignment(viewer, default="auto" if name == "view3d" else "start")
        if name == "view3d":
            viewer.add_argument("--slices", action="store_true", help="start with 2D slices")
            viewer.add_argument("--render", choices=("mip", "attenuated_mip", "minip", "average"), default="mip")
            viewer.add_argument("--screenshot", metavar="PATH", help="save a screenshot")

    for name, description in (("tiff", "Export an HT or fluorescence TIFF stack."),
                              ("mat", "Export calibrated arrays and metadata to MATLAB."),
                              ("gif", "Export an HT, fluorescence, or overlay animation."),
                              ("png", "Export a normalized PNG sequence along Z.")):
        exporter = command(name, description, _export)
        _add_selection(exporter)
        exporter.add_argument("output", nargs="?", help="output path (default: source stem and selection)")
        exporter.add_argument("--fl", type=_channel_name, metavar="CHANNEL", help="FL channel (GIF overlay default: CH0)")
        exporter.add_argument("--registered", action="store_true", help="resample selected fluorescence onto the HT grid")
        _add_alignment(exporter, default=None)
        if name == "tiff":
            depth = exporter.add_mutually_exclusive_group()
            depth.add_argument("--16bit", dest="bit_depth", action="store_const", const=16)
            depth.add_argument("--32bit", dest="bit_depth", action="store_const", const=32)
            exporter.set_defaults(bit_depth=32)
            exporter.add_argument("--normalize", action="store_true", help="normalize for visualization; required for 16-bit")
        elif name == "mat":
            exporter.add_argument("--no-fl", action="store_true", help="exclude fluorescence arrays")
        elif name == "gif":
            exporter.add_argument("--overlay", action="store_true", help="blend HT with fluorescence")
            exporter.add_argument("--fps", type=_frame_rate, default=10, help="1-100 fps, rounded to 10 ms intervals")
            exporter.add_argument("--axis", choices=("z", "y", "x"), default="z")
        elif name == "png":
            exporter.add_argument("--prefix", type=_filename_prefix, help="filename prefix (default: channel name)")
            exporter.add_argument("--cmap", default="gray", help="Matplotlib colormap (default: gray)")
            exporter.add_argument("--vmin", type=_finite_number, help="lower display bound (default: 1st percentile)")
            exporter.add_argument("--vmax", type=_finite_number, help="upper display bound (default: 99th percentile)")

    register = command("register", "Estimate and save image-based residual FL translation; reject weak or ambiguous matches.", _register)
    _add_selection(register)
    register.add_argument("output", nargs="?", help="alignment JSON (default: source and channel stem)")
    register.add_argument("--fl", type=_channel_name, required=True, metavar="CHANNEL")
    register.add_argument("--z-offset-mode", choices=("start", "center", "auto"), default="start",
                          help="initial placement before image matching (default: start)")
    register.add_argument("--max-shift", nargs=3, type=_finite_number, default=(10, 10, 10), metavar=("Z", "Y", "X"),
                          help="positive per-axis search bounds in micrometers (default: 10 10 10)")
    register.add_argument("--max-dimension", type=_positive_integer, default=64, help="coarse grid limit per axis, 8-128 (default: 64)")
    register.add_argument("--min-score", type=_finite_number, default=0.6)
    register.add_argument("--min-peak-margin", type=_finite_number, default=0.03)
    register.add_argument("--min-overlap", type=_finite_number, default=0.5)
    register.add_argument("--overwrite", action="store_true", help="replace an existing alignment report")

    analyze = command("analyze", "Measure native calibrated volumes in bounded Z blocks; batch inputs are explicit paths.", _analyze)
    analyze.add_argument("files", nargs="+", metavar="file.TCF")
    selection = analyze.add_mutually_exclusive_group()
    selection.add_argument("--timepoint", type=_nonnegative_integer, metavar="N",
                           help="zero-based acquisition index (default: 0)")
    selection.add_argument("--all-timepoints", action="store_true", help="measure every HT acquisition")
    analyze.add_argument("--channel", action="append", type=_channel_name, metavar="CHANNEL",
                         help="HT or fluorescence channel; repeat to select several (default: HT)")
    analyze.add_argument("--roi", nargs=6, type=_nonnegative_integer,
                         metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                         help="half-open ROI in each channel's native voxel indices")
    analyze.add_argument("--threshold", type=_finite_number, metavar="N",
                         help="inclusive lower threshold in physical RI or fluorescence counts")
    analyze.add_argument("--block-depth", type=_positive_integer, default=16, metavar="N")
    analyze.add_argument("--format", choices=("json", "csv"), default="json")
    analyze.add_argument("--output", metavar="PATH", help="atomically write the report instead of stdout")

    command("version", "Show the installed version.", lambda args: print(f"tomocube-tools v{__version__}") or 0)
    help_parser = command("help", "Show general or command-specific help.", lambda args: 0)
    help_parser.add_argument("topic", nargs="?", choices=tuple(commands.choices), help="command to describe")
    help_parser.set_defaults(handler=lambda args: (commands.choices[args.topic] if args.topic else parser).print_help() or 0)
    return parser


def _validate_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> None:
    if options.command == "tiff" and options.bit_depth == 16 and not options.normalize:
        parser.error("--16bit requires --normalize; use --32bit to preserve physical values")
    if getattr(options, "registration", None) and not options.fl:
        parser.error("--registration requires --fl")
    if options.command in ("tiff", "mat", "gif", "png"):
        registered = options.registered or options.registration is not None
        overlay = options.command == "gif" and options.overlay
        if registered and not options.fl:
            parser.error("--registered requires --fl")
        if options.z_offset_mode is not None and not (registered or overlay):
            parser.error("--z-offset-mode requires --registered or --overlay")
        if options.command == "mat" and options.no_fl and (options.fl or registered):
            parser.error("--no-fl cannot be combined with fluorescence selection or registration")
    if options.command == "register":
        if any(value <= 0 for value in options.max_shift):
            parser.error("--max-shift values must be positive")
        if not 8 <= options.max_dimension <= 128:
            parser.error("--max-dimension must be between 8 and 128")
        if any(not 0 < value <= 1 for value in (options.min_score, options.min_peak_margin, options.min_overlap)):
            parser.error("registration score, peak margin and overlap thresholds must be in (0, 1]")
    if options.command == "png":
        from matplotlib import colormaps

        if options.cmap not in colormaps:
            parser.error(f"unknown colormap: {options.cmap}")
        if options.vmin is not None and options.vmax is not None and options.vmin > options.vmax:
            parser.error("--vmin must be less than or equal to --vmax")
    if options.command == "analyze":
        if options.channel:
            channels = ["HT" if ch.lower() == "ht" else ch for ch in options.channel]
            if len(set(channels)) != len(channels):
                parser.error("--channel selections must not contain duplicates")
        if options.roi is not None and any(start >= stop for start, stop in zip(options.roi[::2], options.roi[1::2])):
            parser.error("--roi requires start < stop for each half-open axis interval")


def _info(options: argparse.Namespace) -> int:
    from tomocube.processing.analysis import inspect_acquisition
    from tomocube.processing.metadata import discover_related_metadata

    inventory = inspect_acquisition(options.file)
    inventory["related_metadata"] = discover_related_metadata(options.file)
    if options.json:
        print(json.dumps(inventory, indent=2, allow_nan=False))
        return 0

    print(f"Source: {inventory['source']}")
    print(f"Acquisitions: {', '.join(inventory['timepoints'])}")
    print(f"Fluorescence channels: {', '.join(inventory['fl_channels']) or 'none'}")
    for volume in inventory["volumes"]:
        label = f"{volume['channel']} / acquisition {volume['acquisition_key']}"
        if volume["status"] == "missing":
            print(f"  {label}: missing")
            continue
        shape, spacing = volume["shape_zyx"], volume["spacing_zyx_um"]
        fov = [size * step for size, step in zip(shape, spacing)]
        print(f"  {label}: shape {shape} ZYX; {volume['dtype']}; {volume['value_unit']}")
        print(f"    spacing {spacing} um; FOV {fov} um; {volume['size_bytes']} bytes")
    if inventory.get("metadata"):
        print("Metadata:")
        print(json.dumps(inventory["metadata"], indent=2, allow_nan=False))
    if inventory["related_metadata"]:
        print("Related metadata (.experiment, .vessel, profile/*.prf):")
        print(json.dumps(inventory["related_metadata"], indent=2, allow_nan=False))
    return 0


def _view(options: argparse.Namespace) -> int:
    from tomocube.viewer import SliceViewer, TCFViewer

    viewer_type = TCFViewer if options.command == "view" else SliceViewer
    with redirect_stdout(sys.stderr):
        with viewer_type(options.file, timepoint=options.timepoint, fl_channel=options.fl,
                         z_offset_mode=options.z_offset_mode, registration_path=options.registration) as viewer:
            viewer.show()
    return 0


def _view3d(options: argparse.Namespace) -> int:
    from tomocube.viewer.viewer_3d import view_3d

    with redirect_stdout(sys.stderr):
        view_3d(options.file, timepoint=options.timepoint, fl_channel=options.fl,
                show_slices=options.slices, rendering=options.render,
                screenshot=options.screenshot, z_offset_mode=options.z_offset_mode,
                registration_path=options.registration)
    return 0


def _export_path(options: argparse.Namespace) -> Path:
    if options.output:
        return Path(options.output)
    stem = Path(options.file).stem
    if options.timepoint:
        stem += f"_t{options.timepoint}"
    if options.registered or options.registration:
        stem += "_registered"
    if options.command == "mat":
        return Path(stem + ".mat")
    if options.command == "tiff":
        return Path(f"{stem}_{options.fl or 'ht'}.tiff")
    if options.command == "png":
        return Path(f"{stem}_{options.fl or 'ht'}_png")
    if options.fl:
        stem += f"_{options.fl}"
    return Path(stem + ("_overlay" if options.overlay else f"_{options.axis}") + ".gif")


def _export(options: argparse.Namespace) -> int:
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.export import export_overlay_gif, export_to_gif, export_to_mat, export_to_png_sequence, export_to_tiff

    output = _export_path(options)
    registration_options = dict(registered=options.registered, z_offset_mode=options.z_offset_mode or "start",
                                registration_path=options.registration)
    if options.command == "mat":
        channels = [] if options.no_fl else [options.fl] if options.fl else None
    elif options.command == "gif" and options.overlay:
        channels = [options.fl or "CH0"]
    else:
        channels = [options.fl] if options.fl else []
    with redirect_stdout(sys.stderr), TCFFileLoader(options.file) as loader:
        loader.load_timepoint(options.timepoint, fl_channels=channels)
        if options.command == "tiff":
            result = export_to_tiff(loader, output, channel=options.fl or "ht",
                                    bit_depth=options.bit_depth, normalize=options.normalize, **registration_options)
        elif options.command == "mat":
            result = export_to_mat(loader, output, include_fl=not options.no_fl,
                                   fl_channel=options.fl, **registration_options)
        elif options.command == "png":
            result = export_to_png_sequence(loader, output, channel=options.fl or "ht",
                                            prefix=options.prefix or "", cmap=options.cmap,
                                            vmin=options.vmin, vmax=options.vmax, **registration_options)
        elif options.overlay:
            result = export_overlay_gif(loader, output, axis=options.axis, fps=options.fps,
                                        fl_channel=options.fl or "CH0", z_offset_mode=options.z_offset_mode or "start",
                                        registration_path=options.registration)
        else:
            result = export_to_gif(loader, output, channel=options.fl or "ht", axis=options.axis,
                                   fps=options.fps, **registration_options)
    for path in result if isinstance(result, list) else [result]:
        print(path)
    return 0


def _register(options: argparse.Namespace) -> int:
    from dataclasses import asdict
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.alignment import estimate_translation, save_alignment

    suffix = f"_t{options.timepoint}" if options.timepoint else ""
    output = options.output or f"{Path(options.file).stem}{suffix}_{options.fl}_registration.json"
    if Path(output).exists() and not options.overwrite:
        raise FileExistsError(f"Output already exists: {output}; choose another path or --overwrite")
    with redirect_stdout(sys.stderr), TCFFileLoader(options.file) as loader:
        loader.load_timepoint(options.timepoint, fl_channels=[options.fl])
        result = estimate_translation(loader.data_3d, loader.fl_data[options.fl], loader.reg_params,
                                      channel=options.fl, z_offset_mode=options.z_offset_mode,
                                      max_shift_um=options.max_shift, max_dimension=options.max_dimension,
                                      min_score=options.min_score, min_peak_margin=options.min_peak_margin,
                                      min_overlap=options.min_overlap)
        save_alignment(loader, options.fl, result, output, overwrite=options.overwrite)
    print(json.dumps({"report": str(output), **asdict(result)}, indent=2, allow_nan=False))
    return 0 if result.accepted else 1


def _analyze(options: argparse.Namespace) -> int:
    from tomocube.processing.analysis import analyze_acquisition
    from tomocube.processing.outputs import atomic_output

    roi = tuple(zip(options.roi[::2], options.roi[1::2])) if options.roi is not None else None
    channels = ["HT" if ch.lower() == "ht" else ch for ch in options.channel] if options.channel else None
    timepoints = None if options.all_timepoints else [options.timepoint if options.timepoint is not None else 0]
    rows = []
    for source in options.files:
        rows.extend(analyze_acquisition(source, timepoints=timepoints, channels=channels,
                                         roi=roi, threshold=options.threshold, block_depth=options.block_depth))
    if options.format == "json":
        report = json.dumps(rows, indent=2, allow_nan=False) + "\n"
    else:
        stream = io.StringIO(newline="")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict, tuple)) else value
                          for key, value in row.items()} for row in rows)
        report = stream.getvalue()
    if options.output:
        with atomic_output(options.output, sources=options.files) as temporary:
            temporary.write_text(report, encoding="utf-8", newline="")
    else:
        sys.stdout.write(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command: 0 success, 1 runtime/data error, 2 invalid arguments."""
    parser = build_parser()
    try:
        options = parser.parse_args(argv)
        if options.command is None:
            parser.print_help()
            return 1
        _validate_options(parser, options)
    except SystemExit as error:
        return int(error.code or 0)

    from tomocube.core.config import get_config
    from tomocube.core.exceptions import TCFError

    config = get_config()
    previous = config.verbose, config.output
    config.verbose, config.output = options.verbose, sys.stderr
    try:
        return options.handler(options)
    except (TCFError, OSError, ValueError, IndexError, ImportError) as error:
        print(f"tomocube: error: {error}", file=sys.stderr)
        return 1
    finally:
        config.verbose, config.output = previous
