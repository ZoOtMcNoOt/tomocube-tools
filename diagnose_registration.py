"""
Diagnostic script to examine TCF file registration metadata.
This will help verify the correct interpretation of FL-to-HT registration.

Creates visual diagnostics to validate:
1. XY alignment (rotation, scale, translation)
2. Z alignment (OffsetZ interpretation)
3. Resolution interpretation
"""

import sys
import h5py
import numpy as np
from pathlib import Path


def diagnose_registration(tcf_path: str, output_dir: str = None):
    """Dump all registration-related metadata from a TCF file."""

    print(f"\n{'='*70}")
    print(f"TCF Registration Diagnostic: {Path(tcf_path).name}")
    print(f"{'='*70}")

    tcf_file = Path(tcf_path)
    if output_dir is None:
        output_dir = tcf_file.parent / "registration_diagnostic"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with h5py.File(tcf_path, 'r') as f:

        # Print HDF5 structure overview
        print("\n[1] HDF5 STRUCTURE OVERVIEW")
        print("-" * 40)
        def print_structure(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
                for attr_name, attr_val in obj.attrs.items():
                    val = np.asarray(attr_val)
                    if val.size == 1:
                        print(f"{indent}  @{attr_name} = {val.item()}")
                    else:
                        print(f"{indent}  @{attr_name} = {val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: {obj.shape} {obj.dtype}")

        f.visititems(print_structure)

        # HT Data
        print("\n[2] HOLOTOMOGRAPHY (HT) DATA")
        print("-" * 40)
        if "Data/3D" in f:
            ht_group = f["Data/3D"]
            print(f"Path: Data/3D")
            for attr_name, attr_val in ht_group.attrs.items():
                val = np.asarray(attr_val)
                print(f"  @{attr_name} = {val.item() if val.size == 1 else val}")

            # Find timepoint dataset
            for key in ht_group.keys():
                ds = ht_group[key]
                if isinstance(ds, h5py.Dataset):
                    print(f"  Dataset '{key}': shape={ds.shape}, dtype={ds.dtype}")
                    ht_shape = ds.shape

        # FL Data
        print("\n[3] FLUORESCENCE (FL) DATA")
        print("-" * 40)
        if "Data/3DFL" in f:
            fl_group = f["Data/3DFL"]
            print(f"Path: Data/3DFL")
            for attr_name, attr_val in fl_group.attrs.items():
                val = np.asarray(attr_val)
                print(f"  @{attr_name} = {val.item() if val.size == 1 else val}")

            for ch_name in fl_group.keys():
                ch = fl_group[ch_name]
                if isinstance(ch, h5py.Group):
                    print(f"\n  Channel: {ch_name}")
                    for attr_name, attr_val in ch.attrs.items():
                        val = np.asarray(attr_val)
                        print(f"    @{attr_name} = {val.item() if val.size == 1 else val}")

                    for ds_name in ch.keys():
                        ds = ch[ds_name]
                        if isinstance(ds, h5py.Dataset):
                            print(f"    Dataset '{ds_name}': shape={ds.shape}, dtype={ds.dtype}")

        # FL Registration
        print("\n[4] FL REGISTRATION PARAMETERS")
        print("-" * 40)
        if "Info/MetaData/FL/Registration" in f:
            reg = f["Info/MetaData/FL/Registration"]
            print(f"Path: Info/MetaData/FL/Registration")
            for attr_name, attr_val in reg.attrs.items():
                val = np.asarray(attr_val)
                print(f"  @{attr_name} = {val.item() if val.size == 1 else val}")
        else:
            print("  No registration data found at Info/MetaData/FL/Registration")

        # Calculate physical dimensions
        print("\n[5] CALCULATED PHYSICAL DIMENSIONS")
        print("-" * 40)

        # Get resolutions
        ht_res_x = ht_res_y = ht_res_z = None
        fl_res_x = fl_res_y = fl_res_z = None

        if "Data/3D" in f:
            ht = f["Data/3D"]
            ht_res_x = float(np.asarray(ht.attrs.get("ResolutionX", [0.196]))[0])
            ht_res_y = float(np.asarray(ht.attrs.get("ResolutionY", [0.196]))[0])
            ht_res_z = float(np.asarray(ht.attrs.get("ResolutionZ", [0.839]))[0])
            print(f"HT Resolution: X={ht_res_x:.4f}, Y={ht_res_y:.4f}, Z={ht_res_z:.4f} um/px")

        if "Data/3DFL" in f:
            fl = f["Data/3DFL"]
            fl_res_x = float(np.asarray(fl.attrs.get("ResolutionX", [0.122]))[0])
            fl_res_y = float(np.asarray(fl.attrs.get("ResolutionY", [0.122]))[0])
            fl_res_z = float(np.asarray(fl.attrs.get("ResolutionZ", [1.044]))[0])
            print(f"FL Resolution: X={fl_res_x:.4f}, Y={fl_res_y:.4f}, Z={fl_res_z:.4f} um/px")

        print()

        # HT physical range
        if "Data/3D" in f:
            ht_group = f["Data/3D"]
            for key in ht_group.keys():
                if isinstance(ht_group[key], h5py.Dataset):
                    ht_z, ht_y, ht_x = ht_group[key].shape
                    ht_z_um = ht_z * ht_res_z
                    print(f"HT Volume: {ht_z} x {ht_y} x {ht_x} voxels")
                    print(f"HT Z range: slice 0 = 0.00 um, slice {ht_z-1} = {(ht_z-1)*ht_res_z:.2f} um")
                    print(f"HT total Z extent: {ht_z_um:.2f} um")
                    break

        print()

        # FL physical range per channel
        if "Data/3DFL" in f:
            fl_group = f["Data/3DFL"]
            for ch_name in fl_group.keys():
                ch = fl_group[ch_name]
                if isinstance(ch, h5py.Group):
                    offset_z = float(np.asarray(ch.attrs.get("OffsetZ", [0.0]))[0])

                    for ds_name in ch.keys():
                        ds = ch[ds_name]
                        if isinstance(ds, h5py.Dataset):
                            fl_z, fl_y, fl_x = ds.shape
                            fl_z_um = fl_z * fl_res_z

                            print(f"{ch_name} Volume: {fl_z} x {fl_y} x {fl_x} voxels")
                            print(f"{ch_name} OffsetZ from file: {offset_z:.2f} um")
                            print(f"{ch_name} FL slice range: 0 to {fl_z-1}")
                            print(f"{ch_name} FL total Z extent: {fl_z_um:.2f} um")
                            print()

                            # Current interpretation:
                            print(f"CURRENT INTERPRETATION (OffsetZ = HT Z where FL starts):")
                            print(f"  FL slice 0 is at HT Z = {offset_z:.2f} um")
                            print(f"  FL slice {fl_z-1} is at HT Z = {offset_z + (fl_z-1)*fl_res_z:.2f} um")
                            fl_start_ht_slice = offset_z / ht_res_z
                            fl_end_ht_slice = (offset_z + fl_z_um) / ht_res_z
                            print(f"  Maps to HT slices: {fl_start_ht_slice:.1f} to {fl_end_ht_slice:.1f}")
                            print()

                            # Alternative interpretation 1: OffsetZ is FL's Z=0 position
                            # This is actually what we have

                            # Alternative interpretation 2: OffsetZ is the offset to ADD to FL Z
                            print(f"ALTERNATIVE: What if OffsetZ is added to FL coordinates?")
                            print(f"  FL slice 0 physical Z in FL coords: 0.00 um")
                            print(f"  FL slice 0 in HT coords: 0.00 + {offset_z:.2f} = {offset_z:.2f} um")
                            print(f"  (Same as current interpretation)")
                            print()

                            # Alternative interpretation 3: What if Z=0 should align with FL center?
                            fl_center_z_um = (fl_z / 2) * fl_res_z
                            print(f"WHAT IF FL and HT centers should align?")
                            print(f"  FL center Z: slice {fl_z//2} = {fl_center_z_um:.2f} um from FL start")
                            print(f"  HT center Z: slice {ht_z//2} = {(ht_z//2)*ht_res_z:.2f} um")
                            center_offset = (ht_z//2)*ht_res_z - fl_center_z_um
                            print(f"  To align centers, FL offset would be: {center_offset:.2f} um")
                            print()

                            break

        print("\n[6] VERIFICATION QUESTIONS")
        print("-" * 40)
        print("1. Does the FL data appear in the correct Z region of the HT volume?")
        print("2. Is the cell/object visible in the same Z slices for both modalities?")
        print("3. When you scroll through Z, does the FL overlay track the HT features?")
        print()
        print("If FL appears shifted relative to HT:")
        print("  - Check if OffsetZ should be SUBTRACTED instead of defining start position")
        print("  - Check if there's an additional offset not captured in metadata")
        print("  - The acquisitions may have different focal planes")

        # Generate visual validation
        print("\n[7] GENERATING VISUAL VALIDATION")
        print("-" * 40)
        generate_visual_validation(f, output_dir)
        print(f"\nDiagnostic images saved to: {output_dir}")


def generate_visual_validation(f: h5py.File, output_dir: Path):
    """Generate images to visually validate registration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Get data
    ht_data = None
    fl_data = None
    ht_res = None
    fl_res = None
    fl_offset_z = 0.0
    reg_params = {}

    # Load HT
    if "Data/3D" in f:
        ht_group = f["Data/3D"]
        for key in ht_group.keys():
            if isinstance(ht_group[key], h5py.Dataset):
                ht_data = np.array(ht_group[key])
                break
        ht_res = {
            'x': float(np.asarray(ht_group.attrs.get("ResolutionX", [0.196]))[0]),
            'y': float(np.asarray(ht_group.attrs.get("ResolutionY", [0.196]))[0]),
            'z': float(np.asarray(ht_group.attrs.get("ResolutionZ", [0.839]))[0]),
        }

    # Load FL
    if "Data/3DFL" in f:
        fl_group = f["Data/3DFL"]
        fl_res = {
            'x': float(np.asarray(fl_group.attrs.get("ResolutionX", [0.122]))[0]),
            'y': float(np.asarray(fl_group.attrs.get("ResolutionY", [0.122]))[0]),
            'z': float(np.asarray(fl_group.attrs.get("ResolutionZ", [1.044]))[0]),
        }
        for ch_name in fl_group.keys():
            ch = fl_group[ch_name]
            if isinstance(ch, h5py.Group):
                fl_offset_z = float(np.asarray(ch.attrs.get("OffsetZ", [0.0]))[0])
                for ds_name in ch.keys():
                    if isinstance(ch[ds_name], h5py.Dataset):
                        fl_data = np.array(ch[ds_name])
                        break
                break

    # Load registration params
    if "Info/MetaData/FL/Registration" in f:
        reg = f["Info/MetaData/FL/Registration"]
        reg_params = {
            'rotation': float(np.asarray(reg.attrs.get("Rotation", [0.0]))[0]),
            'scale': float(np.asarray(reg.attrs.get("Scale", [1.0]))[0]),
            'trans_x': float(np.asarray(reg.attrs.get("TranslationX", [0.0]))[0]),
            'trans_y': float(np.asarray(reg.attrs.get("TranslationY", [0.0]))[0]),
        }

    if ht_data is None:
        print("  No HT data found, skipping visual validation")
        return

    # 1. Z-axis alignment diagram
    print("  Creating Z-axis alignment diagram...")
    create_z_alignment_diagram(ht_data, fl_data, ht_res, fl_res, fl_offset_z, output_dir)

    # 2. XY MIP comparison
    if fl_data is not None:
        print("  Creating XY MIP comparison...")
        create_xy_mip_comparison(ht_data, fl_data, ht_res, fl_res, reg_params, output_dir)

        # 3. Z-slice comparison grid
        print("  Creating Z-slice comparison grid...")
        create_z_slice_comparison(ht_data, fl_data, ht_res, fl_res, fl_offset_z, reg_params, output_dir)

    print("  Done!")


def create_z_alignment_diagram(ht_data, fl_data, ht_res, fl_res, fl_offset_z, output_dir):
    """Create a diagram showing Z axis alignment interpretation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    ht_z = ht_data.shape[0]
    ht_z_extent = ht_z * ht_res['z']

    # HT bar
    ax.barh(0, ht_z_extent, height=0.4, left=0, color='blue', alpha=0.6, label='HT Volume')
    ax.text(ht_z_extent/2, 0, f'HT: {ht_z} slices\n{ht_z_extent:.1f} µm', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # FL bars for different interpretations
    if fl_data is not None:
        fl_z = fl_data.shape[0]
        fl_z_extent = fl_z * fl_res['z']

        # Current interpretation: OffsetZ = start position
        fl_start_current = fl_offset_z
        ax.barh(1, fl_z_extent, height=0.3, left=fl_start_current, color='green', alpha=0.7, label=f'FL (OffsetZ={fl_offset_z:.1f} µm = start)')
        ht_slice_start = fl_offset_z / ht_res['z']
        ht_slice_end = (fl_offset_z + fl_z_extent) / ht_res['z']
        ax.text(fl_start_current + fl_z_extent/2, 1, f'FL: {fl_z} slices, {fl_z_extent:.1f} µm\n→ HT slices {ht_slice_start:.0f}-{ht_slice_end:.0f}', 
                ha='center', va='center', fontsize=9)

        # Alternative: center alignment
        ht_center = ht_z_extent / 2
        fl_center = fl_z_extent / 2
        fl_start_center = ht_center - fl_center
        ax.barh(2, fl_z_extent, height=0.3, left=fl_start_center, color='orange', alpha=0.7, label='FL (center-aligned, ignore OffsetZ)')
        ax.text(fl_start_center + fl_z_extent/2, 2, f'Center-aligned\nHT slices {fl_start_center/ht_res["z"]:.0f}-{(fl_start_center+fl_z_extent)/ht_res["z"]:.0f}', 
                ha='center', va='center', fontsize=9)

        # Show actual signal location in FL
        fl_z_profile = np.sum(fl_data, axis=(1, 2))
        fl_z_profile = fl_z_profile / fl_z_profile.max() if fl_z_profile.max() > 0 else fl_z_profile
        signal_center_slice = np.average(np.arange(fl_z), weights=fl_z_profile + 1e-10)
        signal_center_um = signal_center_slice * fl_res['z']

        # Plot signal profile for current interpretation
        signal_x = fl_offset_z + np.arange(fl_z) * fl_res['z']
        ax.plot(signal_x, 0.8 + fl_z_profile * 0.15, 'g-', linewidth=2, label='FL signal (z-profile)')
        ax.axvline(fl_offset_z + signal_center_um, color='green', linestyle='--', alpha=0.7)
        ax.text(fl_offset_z + signal_center_um, 0.95, f'Signal center\n{signal_center_um:.1f} µm from FL start', fontsize=8, ha='center')

    # Axis labels and legend
    ax.set_xlim(-5, max(ht_z_extent, fl_offset_z + fl_z_extent if fl_data is not None else 0) + 5)
    ax.set_ylim(-0.5, 3)
    ax.set_xlabel('Z position (µm)', fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['HT Volume', 'FL (current interp.)', 'FL (center-aligned)'])
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.axvline(ht_z_extent, color='black', linestyle='-', linewidth=2)
    ax.set_title('Z-Axis Alignment Interpretation\n(Verify: Does FL overlay appear in correct HT slices?)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'z_alignment_diagram.png', dpi=150)
    plt.close()


def create_xy_mip_comparison(ht_data, fl_data, ht_res, fl_res, reg_params, output_dir):
    """Create MIP comparison showing XY alignment."""
    import matplotlib.pyplot as plt
    from scipy import ndimage

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # HT MIP
    ht_mip = np.max(ht_data, axis=0)
    ht_p1, ht_p99 = np.percentile(ht_mip, [1, 99])

    # FL MIP
    fl_mip = np.max(fl_data, axis=0)
    fl_p1, fl_p99 = np.percentile(fl_mip, [1, 99])

    # Raw FL MIP
    axes[0, 0].imshow(ht_mip, cmap='gray', vmin=ht_p1, vmax=ht_p99)
    axes[0, 0].set_title(f'HT MIP\n{ht_data.shape[2]}×{ht_data.shape[1]} px @ {ht_res["x"]:.3f} µm/px')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fl_mip, cmap='Greens', vmin=fl_p1, vmax=fl_p99)
    axes[0, 1].set_title(f'FL MIP (raw)\n{fl_data.shape[2]}×{fl_data.shape[1]} px @ {fl_res["x"]:.3f} µm/px')
    axes[0, 1].axis('off')

    # Apply registration transform to FL MIP
    ht_h, ht_w = ht_data.shape[1], ht_data.shape[2]
    fl_h, fl_w = fl_data.shape[1], fl_data.shape[2]

    # Build transform
    rotation = reg_params.get('rotation', 0)
    scale = reg_params.get('scale', 1.0)
    trans_x = reg_params.get('trans_x', 0)
    trans_y = reg_params.get('trans_y', 0)

    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    # Resolution ratios
    scale_y = fl_res['y'] / ht_res['y']
    scale_x = fl_res['x'] / ht_res['x']

    total_scale_y = scale_y * scale
    total_scale_x = scale_x * scale

    fl_center_y, fl_center_x = fl_h / 2, fl_w / 2
    ht_center_y, ht_center_x = ht_h / 2, ht_w / 2

    # Translation in FL pixels
    trans_y_px = trans_y / fl_res['y']
    trans_x_px = trans_x / fl_res['x']

    R_inv = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    S_inv = np.array([[total_scale_y, 0], [0, total_scale_x]])
    RS = R_inv @ S_inv

    offset = -RS @ np.array([ht_center_y, ht_center_x]) - np.array([trans_y_px, trans_x_px]) + np.array([fl_center_y, fl_center_x])

    fl_registered = ndimage.affine_transform(
        fl_mip.astype(np.float32),
        RS,
        offset=offset,
        output_shape=(ht_h, ht_w),
        order=1,
        mode='constant',
        cval=0.0
    )

    axes[0, 2].imshow(fl_registered, cmap='Greens', vmin=fl_p1, vmax=fl_p99)
    axes[0, 2].set_title(f'FL MIP (registered to HT)\nrot={np.degrees(rotation):.2f}°, scale={scale:.4f}')
    axes[0, 2].axis('off')

    # Overlay
    ht_norm = (ht_mip - ht_p1) / (ht_p99 - ht_p1 + 1e-10)
    ht_norm = np.clip(ht_norm, 0, 1)
    fl_norm = (fl_registered - fl_p1) / (fl_p99 - fl_p1 + 1e-10)
    fl_norm = np.clip(fl_norm, 0, 1)

    overlay = np.zeros((ht_h, ht_w, 3))
    overlay[:, :, 0] = ht_norm  # Red = HT
    overlay[:, :, 1] = fl_norm  # Green = FL
    overlay[:, :, 2] = ht_norm * 0.5  # Blue = slight HT

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (HT=magenta, FL=green)\nYellow = overlap')
    axes[1, 0].axis('off')

    # Show registration params
    params_text = (
        f"Registration Parameters:\n"
        f"  Rotation: {np.degrees(rotation):.4f}°\n"
        f"  Scale: {scale:.6f}\n"
        f"  Translation X: {trans_x:.2f} µm ({trans_x_px:.1f} FL px)\n"
        f"  Translation Y: {trans_y:.2f} µm ({trans_y_px:.1f} FL px)\n\n"
        f"Resolution:\n"
        f"  HT: {ht_res['x']:.4f} × {ht_res['y']:.4f} µm/px\n"
        f"  FL: {fl_res['x']:.4f} × {fl_res['y']:.4f} µm/px\n"
        f"  Ratio: {fl_res['x']/ht_res['x']:.4f}\n\n"
        f"Combined scale factor: {total_scale_x:.4f}"
    )
    axes[1, 1].text(0.1, 0.5, params_text, fontsize=10, family='monospace', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Transform Parameters')

    # Checkerboard comparison
    block_size = 64
    checkerboard = np.zeros((ht_h, ht_w))
    for i in range(0, ht_h, block_size):
        for j in range(0, ht_w, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                checkerboard[i:i+block_size, j:j+block_size] = 1

    checker_img = np.where(checkerboard[..., None] > 0.5, 
                           np.stack([ht_norm]*3, axis=2),
                           np.stack([np.zeros_like(ht_norm), fl_norm, np.zeros_like(ht_norm)], axis=2))
    axes[1, 2].imshow(checker_img)
    axes[1, 2].set_title('Checkerboard (HT gray / FL green)\nEdges should align')
    axes[1, 2].axis('off')

    plt.suptitle('XY Registration Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'xy_registration_comparison.png', dpi=150)
    plt.close()


def create_z_slice_comparison(ht_data, fl_data, ht_res, fl_res, fl_offset_z, reg_params, output_dir):
    """Create comparison of specific Z slices."""
    import matplotlib.pyplot as plt
    from scipy import ndimage

    ht_z = ht_data.shape[0]
    fl_z = fl_data.shape[0]

    # Find which HT slices correspond to FL data
    fl_start_ht_slice = int(fl_offset_z / ht_res['z'])
    fl_z_extent = fl_z * fl_res['z']
    fl_end_ht_slice = int((fl_offset_z + fl_z_extent) / ht_res['z'])

    # Pick slices to compare
    slice_indices = []
    # Before FL starts
    if fl_start_ht_slice > 5:
        slice_indices.append(fl_start_ht_slice - 5)
    # At FL start
    slice_indices.append(min(fl_start_ht_slice, ht_z - 1))
    # Middle of FL
    middle = (fl_start_ht_slice + min(fl_end_ht_slice, ht_z - 1)) // 2
    slice_indices.append(middle)
    # At FL end
    slice_indices.append(min(fl_end_ht_slice, ht_z - 1))
    # After FL ends
    if fl_end_ht_slice < ht_z - 5:
        slice_indices.append(min(fl_end_ht_slice + 5, ht_z - 1))

    slice_indices = sorted(set(slice_indices))

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))

    # Registration transform setup
    ht_h, ht_w = ht_data.shape[1], ht_data.shape[2]
    fl_h, fl_w = fl_data.shape[1], fl_data.shape[2]

    rotation = reg_params.get('rotation', 0)
    scale = reg_params.get('scale', 1.0)
    trans_x = reg_params.get('trans_x', 0)
    trans_y = reg_params.get('trans_y', 0)

    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    scale_y = fl_res['y'] / ht_res['y']
    scale_x = fl_res['x'] / ht_res['x']
    total_scale_y = scale_y * scale
    total_scale_x = scale_x * scale

    fl_center_y, fl_center_x = fl_h / 2, fl_w / 2
    ht_center_y, ht_center_x = ht_h / 2, ht_w / 2
    trans_y_px = trans_y / fl_res['y']
    trans_x_px = trans_x / fl_res['x']

    R_inv = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    S_inv = np.array([[total_scale_y, 0], [0, total_scale_x]])
    RS = R_inv @ S_inv
    offset = -RS @ np.array([ht_center_y, ht_center_x]) - np.array([trans_y_px, trans_x_px]) + np.array([fl_center_y, fl_center_x])

    for col, ht_slice_idx in enumerate(slice_indices):
        ht_slice = ht_data[ht_slice_idx]
        ht_z_um = ht_slice_idx * ht_res['z']
        ht_p1, ht_p99 = np.percentile(ht_slice, [1, 99])

        # Calculate corresponding FL slice
        fl_z_um = ht_z_um - fl_offset_z
        fl_slice_idx = fl_z_um / fl_res['z']

        has_fl = 0 <= fl_slice_idx < fl_z - 1

        # HT slice
        axes[0, col].imshow(ht_slice, cmap='gray', vmin=ht_p1, vmax=ht_p99)
        axes[0, col].set_title(f'HT slice {ht_slice_idx}\nZ = {ht_z_um:.1f} µm')
        axes[0, col].axis('off')

        if has_fl:
            # Interpolate FL slice
            fl_z0 = int(np.floor(fl_slice_idx))
            fl_z1 = min(fl_z0 + 1, fl_z - 1)
            fz = fl_slice_idx - fl_z0
            fl_interp = (1 - fz) * fl_data[fl_z0] + fz * fl_data[fl_z1]

            # Register
            fl_registered = ndimage.affine_transform(
                fl_interp.astype(np.float32),
                RS, offset=offset,
                output_shape=(ht_h, ht_w),
                order=1, mode='constant', cval=0.0
            )

            fl_p1, fl_p99 = np.percentile(fl_registered[fl_registered > 0], [1, 99]) if np.any(fl_registered > 0) else (0, 1)

            axes[1, col].imshow(fl_registered, cmap='Greens', vmin=fl_p1, vmax=fl_p99)
            axes[1, col].set_title(f'FL slice ~{fl_slice_idx:.1f}\n(Z = {fl_z_um:.1f} µm in FL coords)')
            axes[1, col].axis('off')

            # Overlay
            ht_norm = (ht_slice - ht_p1) / (ht_p99 - ht_p1 + 1e-10)
            ht_norm = np.clip(ht_norm, 0, 1)
            fl_norm = (fl_registered - fl_p1) / (fl_p99 - fl_p1 + 1e-10)
            fl_norm = np.clip(fl_norm, 0, 1)

            overlay = np.zeros((ht_h, ht_w, 3))
            overlay[:, :, 0] = ht_norm
            overlay[:, :, 1] = fl_norm
            overlay[:, :, 2] = ht_norm * 0.5

            axes[2, col].imshow(overlay)
            axes[2, col].set_title('Overlay')
            axes[2, col].axis('off')
        else:
            axes[1, col].text(0.5, 0.5, 'No FL data\nat this Z', ha='center', va='center', fontsize=12, transform=axes[1, col].transAxes)
            axes[1, col].set_title(f'FL: outside range\n(would be slice {fl_slice_idx:.1f})')
            axes[1, col].axis('off')
            
            axes[2, col].imshow(ht_slice, cmap='gray', vmin=ht_p1, vmax=ht_p99)
            axes[2, col].set_title('HT only')
            axes[2, col].axis('off')

    # Add row labels
    fig.text(0.02, 0.83, 'HT', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.5, 'FL', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.17, 'Overlay', fontsize=14, fontweight='bold', rotation=90, va='center')

    plt.suptitle(f'Z-Slice Comparison\nFL OffsetZ = {fl_offset_z:.1f} µm → FL spans HT slices {fl_start_ht_slice}-{fl_end_ht_slice}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'z_slice_comparison.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_registration.py <path_to_file.TCF>")
        sys.exit(1)

    diagnose_registration(sys.argv[1])