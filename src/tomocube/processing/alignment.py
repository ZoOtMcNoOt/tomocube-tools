"""Conservative image-based residual translation and acquisition-bound reports.

The objective is overlap-normalized cross correlation of positive contrast
intensities, using scipy.signal.correlate for a bounded global shift search.
This requires shared structures: HT and FL need not provide that correspondence.
Scores and peak separation are diagnostics, not a probability of correctness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize, signal

from tomocube.core.types import RegistrationParams
from tomocube.processing.outputs import atomic_output
from tomocube.processing.registration import FluorescenceRegistration, Z_OFFSET_MODES


@dataclass(frozen=True)
class AlignmentResult:
    """Residual translation in HT-world ZYX micrometers; metadata stays intact."""

    accepted: bool
    reason: str
    translation_um: tuple[float, float, float]
    score_before: float
    score_after: float
    peak_margin: float
    overlap_fraction: float
    z_offset_mode: str
    grid_spacing_um: tuple[float, float, float]
    max_shift_um: tuple[float, float, float]
    min_score: float
    min_peak_margin: float
    min_overlap: float
    max_dimension: int


def _triple(value, name, *, positive=False):
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,) or not np.isfinite(values).all() or positive and (values <= 0).any():
        raise ValueError(f"{name} must contain {'positive ' if positive else ''}finite ZYX values")
    return values


def _standardize(data, valid):
    values = data[valid]
    if values.size == 0:
        return np.zeros_like(data, dtype=float)
    low, high = np.percentile(values, [1, 99.5])
    if high <= low:
        low, high = values.min(), values.max()
    if high <= low:
        return np.zeros_like(data, dtype=float)
    # Keep a single linear intensity relationship, without changing polarity.
    return np.where(valid, (data.astype(float) - low) / (high - low), 0)


def _correlation(reference, moving, valid):
    a, b = reference[valid], moving[valid]
    if a.size < 32:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.clip(np.sum(a * b) / denominator, -1, 1)) if denominator > 1e-12 else 0.0


def estimate_translation(ht_data, fl_data, params: RegistrationParams | None = None, *,
                         channel=None, z_offset_mode="start", max_shift_um=10.0,
                         max_dimension=64, min_score=0.6, min_peak_margin=0.03,
                         min_overlap=0.5) -> AlignmentResult:
    """Estimate translation after metadata rotation/scale/placement.

    The coarse search samples a regular grid of at most max_dimension per axis,
    computes masked local normalized correlation for each non-wrapping shift,
    and refines the best shift against original FL samples. Bounds are per-axis
    micrometers. No rotation, scale, deformation or cross-modal correspondence
    is inferred. Small, flat, ambiguous and low-overlap data return a rejected
    result; invalid data/options raise ValueError. Sources are never modified.
    """
    ht = np.asarray(ht_data)
    if (ht.ndim != 3 or not ht.size or ht.dtype.kind not in "iuf"
            or not np.isfinite(ht).all()):
        raise ValueError("HT data must be a nonempty finite real 3D volume")
    if (not isinstance(max_dimension, (int, np.integer)) or isinstance(max_dimension, bool)
            or not 8 <= max_dimension <= 128):
        raise ValueError("max_dimension must be an integer from 8 to 128")
    bounds = _triple(max_shift_um, "max_shift_um", positive=True)
    for name, value in (("min_score", min_score), ("min_peak_margin", min_peak_margin),
                        ("min_overlap", min_overlap)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    params = params or RegistrationParams()
    registration = FluorescenceRegistration(fl_data, ht.shape, params, channel, z_offset_mode)
    # Bound both the reference and the shift halo. Native FL outside the
    # baseline HT footprint must participate: it can enter after translation.
    strides = np.maximum(1, np.maximum(np.ceil(np.array(ht.shape) / max_dimension),
                         np.ceil(2 * bounds / registration.ht_spacing / max_dimension))).astype(int)
    spacing = registration.ht_spacing * strides
    slices = tuple(slice(None, None, int(step)) for step in strides)
    reference = ht[slices].astype(float)
    grid = np.indices(reference.shape, dtype=float) * strides[:, None, None, None]
    coordinates = np.einsum("ij,j...->i...", registration.matrix, grid)
    coordinates += registration.offset[:, None, None, None]

    def sample(shift, base_coordinates=coordinates):
        coords = base_coordinates - (registration.matrix @ (shift / registration.ht_spacing))[:, None, None, None]
        valid = np.ones(coords.shape[1:], dtype=bool)
        for axis, length in enumerate(registration.data.shape):
            # Same voxel-center boundary convention as the shared plane sampler.
            coords[axis][np.abs(coords[axis]) < 1e-9] = 0
            coords[axis][np.abs(coords[axis] - (length - 1)) < 1e-9] = length - 1
            valid &= (coords[axis] >= 0) & (coords[axis] <= length - 1)
        moving = ndimage.map_coordinates(registration.data, coords, order=1, mode="constant",
                                         cval=0, prefilter=False, output=np.float64)
        return moving, valid

    moving, mask = sample(np.zeros(3))
    initial_count = int(mask.sum())
    reference = _standardize(reference, np.ones(reference.shape, dtype=bool))
    before = _correlation(reference, moving, mask)

    def result(reason, shift=(0, 0, 0), score=0.0, margin=0.0, overlap=0.0):
        return AlignmentResult(reason == "accepted", reason, tuple(float(v) for v in shift),
                               before, float(score), float(margin), float(overlap), z_offset_mode,
                               tuple(float(v) for v in spacing), tuple(float(v) for v in bounds),
                               float(min_score), float(min_peak_margin), float(min_overlap), int(max_dimension))

    if min(reference.shape) < 4:
        return result("insufficient_3d_extent")
    if initial_count < max(32, min_overlap * reference.size):
        return result("insufficient_initial_overlap")
    if reference.std() < 1e-8 or _standardize(moving, mask)[mask].std() < 1e-8:
        return result("insufficient_signal")

    padding = np.ceil(bounds / spacing).astype(int)
    extended_grid = np.indices(np.array(reference.shape) + 2 * padding, dtype=float)
    extended_grid = (extended_grid - padding[:, None, None, None]) * strides[:, None, None, None]
    extended_coordinates = np.einsum("ij,j...->i...", registration.matrix, extended_grid)
    extended_coordinates += registration.offset[:, None, None, None]
    moving, mask = sample(np.zeros(3), extended_coordinates)
    moving = _standardize(moving, mask)
    del extended_grid, extended_coordinates

    # Normalize independently at each overlap, so image borders and empty
    # out-of-volume padding cannot become the registration signal.
    ones = np.ones(reference.shape)
    mask_float = mask.astype(float)
    def correlate(a, b):
        # b contains the halo. Reversing valid correlation orders the residual
        # shifts from -padding to +padding instead of native window positions.
        return signal.correlate(b, a, mode="valid", method="fft")[::-1, ::-1, ::-1]
    count = np.maximum(np.rint(correlate(ones, mask_float)), 1)
    sum_a = correlate(reference, mask_float)
    sum_b = correlate(ones, moving)
    variance_a = np.maximum(correlate(reference * reference, mask_float) - sum_a * sum_a / count, 0)
    variance_b = np.maximum(correlate(ones, moving * moving) - sum_b * sum_b / count, 0)
    denominator = np.sqrt(variance_a * variance_b)
    covariance = correlate(reference, moving) - sum_a * sum_b / count
    scores = np.full(count.shape, -np.inf)
    allowed = (count >= min_overlap * initial_count) & (denominator > 1e-8)
    shifts = [np.arange(-p, p + 1) for p in padding]
    for axis in range(3):
        axis_shape = [1, 1, 1]
        axis_shape[axis] = len(shifts[axis])
        allowed &= (np.abs(shifts[axis] * spacing[axis]) <= bounds[axis]).reshape(axis_shape)
    np.divide(covariance, denominator, out=scores, where=allowed)
    if not np.isfinite(scores).any():
        return result("insufficient_signal")
    peak = np.unravel_index(np.argmax(scores), scores.shape)
    shift = np.array([shifts[axis][peak[axis]] * spacing[axis] for axis in range(3)])
    alternatives = scores.copy()
    alternatives[tuple(slice(max(0, i - 2), i + 3) for i in peak)] = -np.inf
    second = float(np.max(alternatives))

    def evaluate(translation):
        sampled, valid = sample(translation)
        overlap = min(1.0, float(valid.sum() / initial_count))
        return _correlation(reference, sampled, valid), overlap

    def objective(translation):
        score, overlap = evaluate(translation)
        return -score if overlap >= min_overlap else 2.0

    # Bounded derivative-free refinement handles interpolation's piecewise
    # linear objective without gradients or a second resampled FL volume.
    lower, upper = np.maximum(-bounds, shift - spacing), np.minimum(bounds, shift + spacing)
    optimized = optimize.minimize(objective, shift, method="Powell", bounds=list(zip(lower, upper)),
                                   options={"xtol": 0.005, "ftol": 1e-5, "maxiter": 40})
    if optimized.success and objective(optimized.x) < objective(shift):
        shift = optimized.x
    score, overlap = evaluate(shift)
    # Test observability on every axis using actual physical search bounds.
    # A coarse grid face can lie inside the refined peak's basin, so it must
    # not be mistaken for a distinct competitor or for the physical boundary.
    for axis in range(3):
        for direction in (-1, 1):
            candidate = shift.copy()
            candidate[axis] = direction * bounds[axis]
            boundary_score, boundary_overlap = evaluate(candidate)
            if boundary_overlap >= min_overlap:
                second = max(second, boundary_score)
    margin = max(0.0, score - second) if np.isfinite(second) else 0.0
    if score < min_score:
        reason = "weak_correlation"
    elif margin < min_peak_margin:
        reason = "ambiguous_peak"
    elif overlap < min_overlap:
        reason = "insufficient_overlap"
    elif (np.abs(shift) >= bounds - 0.05 * spacing).any():
        reason = "search_boundary"
    elif score + 1e-5 < before:
        reason = "no_improvement"
    else:
        reason = "accepted"
    return result(reason, shift, score, margin, overlap)


def _digest(data):
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(str(data.shape).encode("ascii"))
    for plane in data:
        digest.update(np.ascontiguousarray(plane).tobytes())
    return digest.hexdigest()


def _binding(loader, channel):
    if channel not in loader.fl_data:
        raise ValueError(f"FL channel {channel!r} is not loaded at this acquisition")
    return {"source": loader.tcf_path.name, "timepoint": loader.current_timepoint, "channel": channel,
            "ht_shape": list(loader.data_3d.shape), "fl_shape": list(loader.fl_data[channel].shape),
            "ht_sha256": _digest(loader.data_3d), "fl_sha256": _digest(loader.fl_data[channel]),
            "calibration": asdict(loader.reg_params)}


def save_alignment(loader, channel, result: AlignmentResult, path, *, overwrite=False) -> Path:
    """Save accepted OR rejected diagnostics without modifying source metadata."""
    from tomocube import __version__
    import scipy

    registration = FluorescenceRegistration(loader.fl_data[channel], loader.data_3d.shape,
                                            loader.reg_params, channel, result.z_offset_mode,
                                            translation_um=result.translation_um)
    report = {"schema": "tomocube-alignment-v1", "axis_order": "ZYX", "length_unit": "um",
              "method": "masked-normalized-correlation-translation",
              "software": {"tomocube": __version__, "numpy": np.__version__, "scipy": scipy.__version__},
              "binding": _binding(loader, channel), "result": asdict(result),
              "voxel_to_world": registration.voxel_to_world.tolist()}
    with atomic_output(path, sources=[loader.tcf_path], overwrite=overwrite) as temporary:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return Path(path)


def load_alignment(path, loader, channel) -> AlignmentResult:
    """Validate a successful report against source pixels, calibration and selection."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") != "tomocube-alignment-v1"
            or report.get("axis_order") != "ZYX" or report.get("length_unit") != "um"
            or report.get("method") != "masked-normalized-correlation-translation"):
        raise ValueError("Unsupported alignment report schema or units")
    try:
        result = AlignmentResult(**report["result"])
        if any(np.shape(values) != (3,) for values in
               (result.translation_um, result.max_shift_um, result.grid_spacing_um)):
            raise ValueError("Serialized alignment vectors require three ZYX values")
        translation = _triple(result.translation_um, "translation_um")
        bounds = _triple(result.max_shift_um, "max_shift_um", positive=True)
        _triple(result.grid_spacing_um, "grid_spacing_um", positive=True)
        metrics = [result.score_before, result.score_after, result.peak_margin, result.overlap_fraction,
                   result.min_score, result.min_peak_margin, result.min_overlap]
        if (result.z_offset_mode not in Z_OFFSET_MODES or not np.isfinite(metrics).all()
                or not isinstance(result.max_dimension, int) or isinstance(result.max_dimension, bool)
                or not 8 <= result.max_dimension <= 128
                or (np.abs(translation) > bounds).any()
                or not -1 <= result.score_before <= 1 or not -1 <= result.score_after <= 1
                or not 0 <= result.overlap_fraction <= 1
                or any(not 0 < value <= 1 for value in metrics[-3:])):
            raise ValueError("Invalid alignment parameters or diagnostics")
        if (result.accepted is not True or result.reason != "accepted"
                or result.score_after < result.min_score or result.peak_margin < result.min_peak_margin
                or result.overlap_fraction < result.min_overlap
                or result.score_after + 1e-5 < result.score_before
                or (np.abs(translation) >= bounds - 0.05 * np.asarray(result.grid_spacing_um)).any()):
            raise ValueError("Alignment was rejected; inspect its diagnostics before choosing another approach")
        if report.get("binding") != _binding(loader, channel):
            raise ValueError("Alignment does not match this source, acquisition, channel or calibration")
        expected = FluorescenceRegistration(loader.fl_data[channel], loader.data_3d.shape,
                                            loader.reg_params, channel, result.z_offset_mode,
                                            translation_um=translation).voxel_to_world
        supplied = np.asarray(report["voxel_to_world"], dtype=float)
        if supplied.shape != (4, 4) or not np.allclose(expected, supplied, rtol=1e-10, atol=1e-10):
            raise ValueError("Alignment affine does not match its parameters")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed alignment report") from error
    return result
