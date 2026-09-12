"""Reproduce a synthetic calibrated alignment, analysis and export workflow.

Run: python examples/registration_workflow.py [new-output-directory]
No experimental data is used. The output directory must not already exist.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np

from tomocube import (SliceViewer, TCFFileLoader, analyze_acquisition, estimate_translation,
                      export_to_tiff, inspect_acquisition, save_alignment)


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    shape = (24, 32, 36)
    spacing = np.array((1.2, 0.7, 0.5))
    true_shift = np.array((1.1, -0.8, 0.65))
    coordinates = np.indices(shape) * spacing[:, None, None, None]
    rng = np.random.default_rng(72)
    centers = rng.uniform(0.2, 0.8, (22, 3)) * ((np.array(shape) - 1) * spacing)
    widths, heights = rng.uniform(0.7, 1.5, (22, 3)), rng.uniform(0.5, 1.5, 22)

    def field(grid):
        result = np.zeros(shape)
        for center, width, height in zip(centers, widths, heights):
            distance = (grid - center[:, None, None, None]) / width[:, None, None, None]
            result += height * np.exp(-0.5 * np.sum(distance * distance, axis=0))
        return result

    ht = (1.33 + 0.04 * field(coordinates)).astype(np.float32)
    fl = (100 + 1200 * field(coordinates + true_shift[:, None, None, None])).astype(np.uint16)
    source = output / "synthetic.TCF"
    with h5py.File(source, "w") as file:
        for path, data in (("Data/3D/0", ht), ("Data/3DFL/CH0/0", fl)):
            file.create_dataset(path, data=data)
        for modality in ("Data/3D", "Data/3DFL"):
            for axis, value in zip("ZYX", spacing):
                file[modality].attrs[f"Resolution{axis}"] = value
        file["Data/3DFL/CH0"].attrs["OffsetZ"] = 0.0
    report = output / "alignment.json"
    with TCFFileLoader(source) as loader:
        loader.load_timepoint(0, fl_channels=["CH0"])
        estimate = estimate_translation(ht, fl, loader.reg_params, channel="CH0", max_shift_um=4)
        save_alignment(loader, "CH0", estimate, report)
        if not estimate.accepted:
            raise RuntimeError(f"Synthetic reference estimate rejected: {estimate.reason}")
        export_to_tiff(loader, output / "registered.tiff", channel="CH0", registration_path=report)
    with SliceViewer(source, fl_channel="CH0", registration_path=report) as viewer:
        viewer.fig.savefig(output / "viewer.png", dpi=150)
    evidence = {"synthetic": True, "known_translation_zyx_um": true_shift.tolist(),
                "absolute_error_um": np.abs(np.asarray(estimate.translation_um) - true_shift).tolist(),
                "estimate": asdict(estimate), "inventory": inspect_acquisition(source),
                "analysis": analyze_acquisition(source, channels=["HT", "CH0"], block_depth=4)}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "absolute_error_um": evidence["absolute_error_um"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("output/toolset-qa"))
    run(parser.parse_args().output)
