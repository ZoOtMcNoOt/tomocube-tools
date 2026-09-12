"""
TCF Export - Convert TCF files to various formats.

Supported formats:
    - TIFF stack (multi-page TIFF, 16-bit or 32-bit)
    - MAT file (MATLAB .mat format)
    - GIF animation (Z-stack or time-lapse)
    - PNG sequence (individual slices)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tomocube.processing.image import normalize_with_bounds

if TYPE_CHECKING:
    from tomocube.core.file import TCFFileLoader


def _gif_duration(fps: int) -> int:
    """Validate the rate and round to GIF's 10 ms frame duration units."""
    if not np.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError("fps must be greater than 0 and at most 100")
    return max(10, round(100 / fps) * 10)


def export_to_tiff(
    loader: TCFFileLoader,
    output_path: str | Path,
    channel: str = "ht",
    bit_depth: int = 16,
    normalize: bool = True,
    compression: str = "lzw",
) -> Path:
    """
    Export TCF data to a multi-page TIFF stack.

    Args:
        loader: TCFFileLoader with loaded data
        output_path: Output file path (will add .tiff if needed)
        channel: "ht" for holotomography, or FL channel name like "CH0"
        bit_depth: 16 or 32 bit output
        normalize: If True, normalize to full bit range (good for visualization).
                   If False with 32-bit, preserves physical RI values.
        compression: TIFF compression ("lzw", "zlib", "none")

    Returns:
        Path to the saved TIFF file

    Note:
        For HT data, physical RI values (e.g., 1.33-1.40) are preserved when
        using bit_depth=32 with normalize=False. This is recommended for
        scientific analysis. For 16-bit output, normalization is required
        to map values to the 0-65535 range.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile package required: pip install tifffile")

    output_path = Path(output_path)
    if not output_path.suffix.lower() in (".tif", ".tiff"):
        output_path = output_path.with_suffix(".tiff")

    if bit_depth not in (16, 32):
        raise ValueError(f"bit_depth must be 16 or 32, got {bit_depth}")
    if bit_depth == 16 and not normalize:
        raise ValueError("16-bit TIFF output requires normalize=True; use 32-bit to preserve values")
    compression_map = {"lzw": "lzw", "zlib": "zlib", "none": None}
    if compression not in compression_map:
        raise ValueError(f"compression must be one of {list(compression_map)}, got {compression!r}")

    # Get data (already in physical RI units for HT)
    if channel.lower() == "ht":
        data = loader.data_3d
        description = f"HT data from {loader.tcf_path.name} (RI values)"
    else:
        if channel not in loader.fl_data:
            raise ValueError(f"FL channel '{channel}' not found. Available: {list(loader.fl_data.keys())}")
        data = loader.fl_data[channel]
        description = f"FL {channel} from {loader.tcf_path.name}"

    description += f"; timepoint {loader.current_timepoint}"

    # Handle conversion based on bit depth and normalization
    if bit_depth == 32:
        if normalize:
            # 32-bit normalized (0-1 range)
            vmin, vmax = np.percentile(data, [0.1, 99.9])
            data_out = normalize_with_bounds(data, vmin, vmax).astype(np.float32)
            description += f" [normalized 0-1, original range {vmin:.4f}-{vmax:.4f}]"
        else:
            # 32-bit preserving physical values (recommended for scientific use)
            data_out = data.astype(np.float32)
            units = "physical RI" if channel.lower() == "ht" else "fluorescence intensity"
            description += f" [{units}, range {data.min():.4f}-{data.max():.4f}]"
    elif bit_depth == 16:
        # 16-bit always requires normalization to map to 0-65535
        vmin, vmax = np.percentile(data, [0.1, 99.9])
        data_norm = normalize_with_bounds(data, vmin, vmax)
        data_out = (data_norm * 65535).astype(np.uint16)
        description += f" [normalized, original range {vmin:.4f}-{vmax:.4f}]"

    # Get resolution for metadata
    res_z, res_y, res_x = (
        loader.tcf_info.ht_resolution if channel.lower() == "ht" else loader.tcf_info.fl_resolution
    )

    # Save with ImageJ-compatible metadata
    tifffile.imwrite(
        output_path,
        data_out,
        imagej=True,
        compression=compression_map[compression],
        metadata={
            "axes": "ZYX",
            "unit": "um",
            "spacing": res_z,
            "Info": description,
        },
        resolution=(1 / res_x, 1 / res_y),
        resolutionunit="NONE",
    )

    return output_path


def export_to_mat(
    loader: TCFFileLoader,
    output_path: str | Path,
    include_fl: bool = True,
    include_metadata: bool = True,
) -> Path:
    """
    Export TCF data to MATLAB .mat format.

    Args:
        loader: TCFFileLoader with loaded data
        output_path: Output file path (will add .mat if needed)
        include_fl: Include fluorescence data if available
        include_metadata: Include resolution and file metadata

    Returns:
        Path to the saved MAT file

    Note:
        HT data is stored in physical refractive index units (e.g., 1.33-1.40).
        Resolution values are in micrometers (μm).

    Variables saved:
        - ht_3d: 3D HT volume (Z, Y, X) in physical RI units
        - ht_mip: Maximum intensity projection
        - fl_ch0, fl_ch1, ...: FL channels (if include_fl=True)
        - metadata: File info (if include_metadata=True)
        - resolution: Spatial resolution in μm (if include_metadata=True)
    """
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy package required: pip install scipy")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mat":
        output_path = output_path.with_suffix(".mat")

    # Build data dictionary
    # HT data is already in physical RI units (e.g., 1.3300 not 13300)
    mat_dict = {
        "ht_3d": loader.data_3d,
        "ht_mip": loader.data_mip,
    }

    # Add FL data
    if include_fl and loader.has_fluorescence:
        for ch_name, ch_data in loader.fl_data.items():
            mat_dict[f"fl_{ch_name.lower()}"] = ch_data

    # Add metadata
    if include_metadata:
        info = loader.tcf_info
        params = loader.reg_params
        metadata = {
            "filename": str(loader.tcf_path.name),
            "timepoint": loader.current_timepoint,
            "ht_shape": loader.data_3d.shape,
            "ht_resolution_um": info.ht_resolution,
            "magnification": info.magnification,
            "numerical_aperture": info.numerical_aperture,
            "medium_ri": info.medium_ri,
            "has_fluorescence": info.has_fluorescence,
        }
        # MATLAB structs cannot encode Python None. Omit unavailable values
        # rather than inventing optical parameters for incomplete metadata.
        mat_dict["metadata"] = {key: value for key, value in metadata.items() if value is not None}
        mat_dict["resolution"] = {
            "ht_res_x_um": params.ht_res_x,
            "ht_res_y_um": params.ht_res_y,
            "ht_res_z_um": params.ht_res_z,
            "fl_res_x_um": params.fl_res_x,
            "fl_res_y_um": params.fl_res_y,
            "fl_res_z_um": params.fl_res_z,
            "fl_offset_z_um": params.fl_offset_z,
        }

    savemat(output_path, mat_dict, do_compression=True)
    return output_path


def export_to_png_sequence(
    loader: TCFFileLoader,
    output_dir: str | Path,
    channel: str = "ht",
    prefix: str = "",
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    """
    Export TCF data as a sequence of PNG images.

    Args:
        loader: TCFFileLoader with loaded data
        output_dir: Output directory
        channel: "ht" for holotomography, or FL channel name
        prefix: Filename prefix (default: use channel name)
        cmap: Matplotlib colormap name
        vmin, vmax: Value range for normalization

    Returns:
        List of paths to saved PNG files
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if channel.lower() == "ht":
        data = loader.data_3d
        prefix = prefix or "ht"
    else:
        if channel not in loader.fl_data:
            raise ValueError(f"FL channel '{channel}' not found")
        data = loader.fl_data[channel]
        prefix = prefix or f"fl_{channel.lower()}"

    if vmin is None or vmax is None:
        p_vmin, p_vmax = np.percentile(data, [1, 99])
        vmin = vmin if vmin is not None else p_vmin
        vmax = vmax if vmax is not None else p_vmax

    saved_files = []
    for z in range(data.shape[0]):
        filename = output_dir / f"{prefix}_{z:04d}.png"
        plt.imsave(filename, data[z], cmap=cmap, vmin=vmin, vmax=vmax)
        saved_files.append(filename)

    return saved_files


def export_to_gif(
    loader: TCFFileLoader,
    output_path: str | Path,
    channel: str = "ht",
    axis: str = "z",
    fps: int = 10,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    loop: int = 0,
) -> Path:
    """
    Export TCF data as an animated GIF.

    Args:
        loader: TCFFileLoader with loaded data
        output_path: Output file path
        channel: "ht" for holotomography, or FL channel name
        axis: Animation axis ("z", "y", or "x")
        fps: Frames per second, greater than 0 and at most 100 (rounded to 10 ms)
        cmap: Matplotlib colormap name
        vmin, vmax: Value range for normalization
        loop: Number of loops (0 = infinite)

    Returns:
        Path to the saved GIF file
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow package required: pip install Pillow")

    from matplotlib import colormaps

    duration_ms = _gif_duration(fps)

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")

    if channel.lower() == "ht":
        data = loader.data_3d
    else:
        if channel not in loader.fl_data:
            raise ValueError(f"FL channel '{channel}' not found")
        data = loader.fl_data[channel]

    # Get slices along the specified axis
    if axis.lower() == "z":
        slices = [data[i] for i in range(data.shape[0])]
    elif axis.lower() == "y":
        slices = [data[:, i, :] for i in range(data.shape[1])]
    elif axis.lower() == "x":
        slices = [data[:, :, i] for i in range(data.shape[2])]
    else:
        raise ValueError(f"axis must be 'z', 'y', or 'x', got '{axis}'")

    if vmin is None or vmax is None:
        p_vmin, p_vmax = np.percentile(data, [1, 99])
        vmin = vmin if vmin is not None else p_vmin
        vmax = vmax if vmax is not None else p_vmax

    # Convert to images
    colormap = colormaps[cmap]
    frames = []
    for slice_data in slices:
        normalized = normalize_with_bounds(slice_data, vmin, vmax)
        colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
        frames.append(Image.fromarray(colored))

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    return output_path


def export_overlay_gif(
    loader: TCFFileLoader,
    output_path: str | Path,
    fl_channel: str = "CH0",
    axis: str = "z",
    fps: int = 10,
    fl_alpha: float = 0.5,
    ht_cmap: str = "gray",
    loop: int = 0,
    z_offset_mode: str = "start",
) -> Path:
    """
    Export HT + FL overlay as an animated GIF.

    Args:
        loader: TCFFileLoader with loaded data
        output_path: Output file path
        fl_channel: FL channel name
        axis: Animation axis ("z", "y", or "x")
        fps: Frames per second, greater than 0 and at most 100 (rounded to 10 ms)
        fl_alpha: FL overlay alpha (0-1)
        ht_cmap: Colormap for HT
        loop: Number of loops (0 = infinite)
        z_offset_mode: How to interpret fl_offset_z:
            - "start": OffsetZ is HT Z position where FL slice 0 starts (default)
            - "center": OffsetZ is HT Z position of FL volume center
            - "auto": Center FL on HT volume, ignoring OffsetZ

    Returns:
        Path to the saved GIF file
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow package required: pip install Pillow")

    from matplotlib import colormaps

    from tomocube.processing.registration import register_fl_to_ht

    duration_ms = _gif_duration(fps)
    if not np.isfinite(fl_alpha) or not 0 <= fl_alpha <= 1:
        raise ValueError("fl_alpha must be between 0 and 1")
    if z_offset_mode not in ("start", "center", "auto"):
        raise ValueError("z_offset_mode must be 'start', 'center', or 'auto'")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")

    if not loader.has_fluorescence:
        raise ValueError("No fluorescence data available")
    if fl_channel not in loader.fl_data:
        raise ValueError(f"FL channel '{fl_channel}' not found")

    ht_data = loader.data_3d
    fl_raw = loader.fl_data[fl_channel]
    fl_registered = register_fl_to_ht(
        fl_raw, ht_data.shape, loader.reg_params,
        channel=fl_channel, z_offset_mode=z_offset_mode
    )

    # Normalize
    ht_vmin, ht_vmax = np.percentile(ht_data, [1, 99])
    fl_nonzero = fl_registered[fl_registered > 0]
    if len(fl_nonzero) > 0:
        fl_vmin, fl_vmax = np.percentile(fl_nonzero, [1, 99])
    else:
        fl_vmin, fl_vmax = 0, 1

    ht_cmap_obj = colormaps[ht_cmap]
    frames = []

    # Get number of slices based on axis
    if axis.lower() == "z":
        num_slices = ht_data.shape[0]
    elif axis.lower() == "y":
        num_slices = ht_data.shape[1]
    elif axis.lower() == "x":
        num_slices = ht_data.shape[2]
    else:
        raise ValueError(f"axis must be 'z', 'y', or 'x', got '{axis}'")

    for i in range(num_slices):
        # Get slices based on axis
        if axis.lower() == "z":
            ht_slice = ht_data[i]
            fl_slice = fl_registered[i]
        elif axis.lower() == "y":
            ht_slice = ht_data[:, i, :]
            fl_slice = fl_registered[:, i, :]
        else:  # x
            ht_slice = ht_data[:, :, i]
            fl_slice = fl_registered[:, :, i]

        # HT slice
        ht_norm = normalize_with_bounds(ht_slice, ht_vmin, ht_vmax)
        ht_rgb = ht_cmap_obj(ht_norm)[:, :, :3]

        # FL slice (green overlay)
        fl_norm = normalize_with_bounds(fl_slice, fl_vmin, fl_vmax)
        fl_rgb = np.zeros((*fl_norm.shape, 3))
        fl_rgb[:, :, 1] = fl_norm

        # Blend
        blended = ht_rgb * (1 - fl_alpha * fl_norm[:, :, np.newaxis]) + fl_rgb * fl_alpha

        frame = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    return output_path
