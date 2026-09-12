import os

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np
import pytest


@pytest.fixture
def make_tcf(tmp_path):
    """Create small, deterministic acquisitions without external sample data."""
    def create(*, scalar_attrs=False, device=True, fluorescence=False, timepoints=None):
        path = tmp_path / "sample acquisition.TCF"
        if timepoints is None:
            timepoints = {"000000": np.arange(60, dtype=np.uint16).reshape(3, 4, 5) + 13300}

        def attribute(value):
            return value if scalar_attrs else [value]

        with h5py.File(path, "w") as f:
            if device:
                f.attrs["DeviceModelType"] = attribute("HTX")
                f.attrs["DeviceSerial"] = attribute("TEST-001")
                f.attrs["SoftwareVersion"] = attribute("1.0")
                dev = f.create_group("Info/Device")
                for name, value in {"Magnification": 60, "NA": 0.8, "RI": 1.33}.items():
                    dev.attrs[name] = attribute(value)
            ht = f.create_group("Data/3D")
            for axis, value in zip("XYZ", (0.25, 0.5, 1.5)):
                ht.attrs[f"Resolution{axis}"] = attribute(value)
            for key, data in timepoints.items():
                ds = ht.create_dataset(key, data=data)
                ds.attrs["RIMin"] = attribute(float(data.min()))
                ds.attrs["RIMax"] = attribute(float(data.max()))
            if fluorescence:
                fl = f.create_group("Data/3DFL")
                for axis, value in zip("XYZ", (0.75, 1.0, 2.0)):
                    fl.attrs[f"Resolution{axis}"] = attribute(value)
                for index, channel in enumerate(("CH0", "CH1")):
                    group = fl.create_group(channel)
                    group.attrs["OffsetZ"] = attribute(float(index))
                    for key, data in timepoints.items():
                        group.create_dataset(key, data=np.arange(data.size, dtype=np.uint16).reshape(data.shape))
                reg = f.create_group("Info/MetaData/FL/Registration")
                for name, value in {"Rotation": 0.0, "TranslationX": 0.0, "TranslationY": 0.0}.items():
                    reg.attrs[name] = attribute(value)
        return path

    return create
