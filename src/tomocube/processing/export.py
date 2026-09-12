"""Scientific volume exports and display images from loaded acquisitions."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tomocube.processing.image import normalize_with_bounds
from tomocube.processing.outputs import atomic_output, new_output_directory
from tomocube.processing.registration import FluorescenceRegistration


def _gif_duration(fps):
    if not np.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError("fps must be greater than 0 and at most 100")
    return max(10, round(100 / fps) * 10)


def _output_path(path, suffix, alternatives=()):
    path = Path(path)
    return path if path.suffix.lower() in (suffix, *alternatives) else path.with_suffix(suffix)


def _source_data(loader, channel):
    if channel.lower() == "ht":
        data, spacing = loader.data_3d, loader.tcf_info.ht_resolution
    else:
        if channel not in loader.fl_data:
            raise ValueError(f"FL channel {channel!r} not found. Available: {list(loader.fl_data)}")
        data, spacing = loader.fl_data[channel], loader.tcf_info.fl_resolution
    if data.dtype.kind not in "iuf" or not np.isfinite(data).all():
        raise ValueError("Export requires finite real intensities")
    return data, spacing


def _registration(loader, channel, z_offset_mode, registration_path):
    if channel.lower() == "ht":
        raise ValueError("Registration requires a fluorescence channel")
    data, _ = _source_data(loader, channel)
    translation = (0, 0, 0)
    estimated = None
    if registration_path is not None:
        from tomocube.processing.alignment import load_alignment
        alignment = load_alignment(registration_path, loader, channel)
        z_offset_mode, translation = alignment.z_offset_mode, alignment.translation_um
        estimated = asdict(alignment)
    mapping = FluorescenceRegistration(data, loader.data_3d.shape, loader.reg_params,
                                       channel, z_offset_mode, translation_um=translation)
    provenance = {"axis_order": "ZYX", "length_unit": "um", "z_offset_mode": z_offset_mode,
                  "interpolation": "linear; zero outside native voxel centers",
                  "voxel_to_world": mapping.voxel_to_world.tolist(),
                  "original_calibration": asdict(loader.reg_params)}
    if estimated is not None:
        provenance["estimate"] = estimated
    return mapping, provenance


def _volume(loader, channel, registered, z_offset_mode, registration_path):
    data, spacing = _source_data(loader, channel)
    provenance = None
    if registered or registration_path is not None:
        mapping, provenance = _registration(loader, channel, z_offset_mode, registration_path)
        data = np.empty(mapping.ht_shape, dtype=np.float32)
        for z in range(mapping.ht_shape[0]):
            data[z] = mapping.sample_plane(0, z)[0]
        spacing = loader.tcf_info.ht_resolution
    return data, spacing, provenance


def _range(data, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        low, high = np.percentile(data, [1, 99])
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    if not np.isfinite([vmin, vmax]).all() or vmin > vmax:
        raise ValueError("Display bounds must be finite with vmin <= vmax")
    return vmin, vmax


def export_to_tiff(loader, output_path, channel="ht", bit_depth=32, normalize=False,
                   compression="lzw", *, registered=False, z_offset_mode="start",
                   registration_path=None) -> Path:
    """Export native or registered ZYX data with independent XYZ calibration.

    API and CLI default to unnormalized float32 values. Registration applies
    only to FL and places it on the HT grid; its original calibration and full
    affine are included in the ImageJ Info JSON. A saved report implies
    registered=True. Sixteen-bit display output requires normalize=True.
    """
    import tifffile

    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    if bit_depth == 16 and not normalize:
        raise ValueError("16-bit TIFF output requires normalize=True; use 32-bit to preserve values")
    compressions = {"lzw": "lzw", "zlib": "zlib", "none": None}
    if compression not in compressions:
        raise ValueError(f"compression must be one of {list(compressions)}")
    output_path = _output_path(output_path, ".tiff", (".tif",))
    data, spacing, registration = _volume(loader, channel, registered, z_offset_mode, registration_path)
    metadata = {"source": loader.tcf_path.name, "timepoint": loader.current_timepoint,
                "channel": channel, "axis_order": "ZYX", "spacing_zyx_um": list(spacing),
                "value_unit": "RI" if channel.lower() == "ht" else "fluorescence intensity",
                "normalized": bool(normalize)}
    if normalize:
        low, high = np.percentile(data, [0.1, 99.9])
        data_out = normalize_with_bounds(data, low, high)
        metadata["original_range"] = [float(low), float(high)]
        data_out = ((data_out * 65535).astype(np.uint16) if bit_depth == 16
                    else data_out.astype(np.float32))
    else:
        limit = np.finfo(np.float32).max
        if data.max() > limit or data.min() < -limit:
            raise ValueError("Intensities cannot be represented as float32")
        data_out = data.astype(np.float32, copy=False)
    if registration is not None:
        metadata["registration"] = registration
    z, y, x = spacing
    sources = [loader.tcf_path] + ([registration_path] if registration_path is not None else [])
    with atomic_output(output_path, sources=sources) as temporary:
        tifffile.imwrite(temporary, data_out, imagej=True, compression=compressions[compression],
                         metadata={"axes": "ZYX", "unit": "um", "spacing": z,
                                   "Info": json.dumps(metadata, allow_nan=False)},
                         resolution=(1 / x, 1 / y), resolutionunit="NONE")
    return output_path


def export_to_mat(loader, output_path, include_fl=True, include_metadata=True, *,
                  fl_channel=None, registered=False, z_offset_mode="start", registration_path=None) -> Path:
    """Export physical RI and native FL, optionally adding one registered FL.

    Registered data is a separate fl_<channel>_registered variable; native
    fluorescence and calibration remain available for reproducible analysis.
    """
    from scipy.io import savemat

    if (registered or registration_path is not None) and (not include_fl or fl_channel is None):
        raise ValueError("Registered MAT export requires include_fl=True and fl_channel")
    if fl_channel is not None and not include_fl:
        raise ValueError("fl_channel requires include_fl=True")
    if fl_channel is not None and fl_channel not in loader.fl_data:
        raise ValueError(f"FL channel {fl_channel!r} not found")
    if not np.isfinite(loader.data_mip).all():
        raise ValueError("MAT export requires a finite HT projection")
    output_path = _output_path(output_path, ".mat")
    data = {"ht_3d": _source_data(loader, "ht")[0], "ht_mip": loader.data_mip}
    if include_fl:
        for channel in ([fl_channel] if fl_channel is not None else loader.fl_data):
            data[f"fl_{channel.lower()}"] = _source_data(loader, channel)[0]
    if registered or registration_path is not None:
        volume, _, registration = _volume(loader, fl_channel, True, z_offset_mode, registration_path)
        data[f"fl_{fl_channel.lower()}_registered"] = volume
        data["registration_json"] = json.dumps(registration, allow_nan=False)
    if include_metadata:
        info, params = loader.tcf_info, loader.reg_params
        metadata = {"filename": loader.tcf_path.name, "timepoint": loader.current_timepoint,
                    "ht_shape": loader.data_3d.shape, "ht_resolution_um": info.ht_resolution,
                    "magnification": info.magnification, "numerical_aperture": info.numerical_aperture,
                    "medium_ri": info.medium_ri, "has_fluorescence": bool(loader.fl_data)}
        data["metadata"] = {key: value for key, value in metadata.items() if value is not None}
        data["resolution"] = {f"{modality}_res_{axis}_um": getattr(params, f"{modality}_res_{axis}")
                              for modality in ("ht", "fl") for axis in "xyz"}
        data["resolution"]["fl_offset_z_um"] = params.fl_offset_z
        data["calibration_json"] = json.dumps(asdict(params), allow_nan=False)
    sources = [loader.tcf_path] + ([registration_path] if registration_path is not None else [])
    with atomic_output(output_path, sources=sources) as temporary:
        savemat(temporary, data, do_compression=True, appendmat=False)
    return output_path


def export_to_png_sequence(loader, output_dir, channel="ht", prefix="", cmap="gray",
                           vmin=None, vmax=None, *, registered=False, z_offset_mode="start",
                           registration_path=None) -> list[Path]:
    """Publish a complete display sequence into a new or empty directory."""
    from matplotlib import colormaps, pyplot as plt

    colormaps[cmap]
    prefix = prefix or ("ht" if channel.lower() == "ht" else f"fl_{channel.lower()}")
    if Path(prefix).name != prefix or any(c in prefix for c in '/\\:') or prefix in (".", ".."):
        raise ValueError("prefix must be a filename component, without path separators")
    data, _, _ = _volume(loader, channel, registered, z_offset_mode, registration_path)
    vmin, vmax = _range(data, vmin, vmax)
    saved = []
    with new_output_directory(output_dir) as temporary:
        for z, plane in enumerate(data):
            name = f"{prefix}_{z:04d}.png"
            plt.imsave(temporary / name, plane, cmap=cmap, vmin=vmin, vmax=vmax)
            saved.append(Path(output_dir) / name)
    return saved


def _axis(axis):
    if not isinstance(axis, str) or axis.lower() not in ("z", "y", "x"):
        raise ValueError("axis must be 'z', 'y', or 'x'")
    return "zyx".index(axis.lower())


def _save_gif(frames, path, duration, loop, sources):
    if not isinstance(loop, (int, np.integer)) or isinstance(loop, bool) or loop < 0:
        raise ValueError("loop must be a nonnegative integer")
    frames = iter(frames)
    first = next(frames)
    with atomic_output(path, sources=sources) as temporary:
        first.save(temporary, save_all=True, append_images=frames, duration=duration,
                   loop=loop, optimize=True)
    return path


def export_to_gif(loader, output_path, channel="ht", axis="z", fps=10, cmap="gray",
                   vmin=None, vmax=None, loop=0, *, registered=False, z_offset_mode="start",
                   registration_path=None) -> Path:
    """Export a display animation along the selected axis."""
    from PIL import Image
    from matplotlib import colormaps

    duration, dimension = _gif_duration(fps), _axis(axis)
    colormap = colormaps[cmap]
    data, _, _ = _volume(loader, channel, registered, z_offset_mode, registration_path)
    vmin, vmax = _range(data, vmin, vmax)
    def frames():
        for index in range(data.shape[dimension]):
            normalized = normalize_with_bounds(np.take(data, index, axis=dimension), vmin, vmax)
            yield Image.fromarray((colormap(normalized)[..., :3] * 255).astype(np.uint8))
    sources = [loader.tcf_path] + ([registration_path] if registration_path is not None else [])
    return _save_gif(frames(), _output_path(output_path, ".gif"), duration, loop, sources)


def export_overlay_gif(loader, output_path, fl_channel="CH0", axis="z", fps=10,
                       fl_alpha=0.5, ht_cmap="gray", loop=0, z_offset_mode="start", *,
                       registration_path=None) -> Path:
    """Export calibrated HT+FL planes without an extra registered 3D volume.

    FL uses the same native positive-intensity contrast suggestion as the 2D
    viewers. A saved report supplies its own base mode and residual translation.
    """
    from PIL import Image
    from matplotlib import colormaps

    duration, dimension = _gif_duration(fps), _axis(axis)
    if not np.isfinite(fl_alpha) or not 0 <= fl_alpha <= 1:
        raise ValueError("fl_alpha must be between 0 and 1")
    colormap = colormaps[ht_cmap]
    mapping, _ = _registration(loader, fl_channel, z_offset_mode, registration_path)
    ht = _source_data(loader, "ht")[0]
    low, high = _range(ht)
    fl_low, fl_high = loader.get_fl_contrast(fl_channel)
    def frames():
        for index in range(ht.shape[dimension]):
            ht_plane = normalize_with_bounds(np.take(ht, index, axis=dimension), low, high)
            fl_plane = normalize_with_bounds(mapping.sample_plane(dimension, index)[0], fl_low, fl_high)
            rgb = colormap(ht_plane)[..., :3] * (1 - fl_alpha * fl_plane[..., None])
            rgb[..., 1] += fl_plane * fl_alpha
            yield Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    sources = [loader.tcf_path] + ([registration_path] if registration_path is not None else [])
    return _save_gif(frames(), _output_path(output_path, ".gif"), duration, loop, sources)
