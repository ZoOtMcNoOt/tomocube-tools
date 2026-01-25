"""
FL to HT Registration Module for Tomocube TCF Files

This module implements the registration transform to align fluorescence (FL) 
data with holotomography (HT) phase data from Tomocube microscopes.

## Key Insight
Both HT and FL images cover the same physical field of view (~230 um).
- HT: 1172 x 1172 pixels @ 0.196 um/pixel = 230 um
- FL: 1893 x 1893 pixels @ 0.122 um/pixel = 230 um

The primary registration is simply resampling FL to match HT pixel dimensions.
Small corrections (rotation, translation, scale) can be applied from metadata.

## Registration Parameters (from TCF metadata):
- Info/MetaData/FL/Registration/Rotation: fine rotation correction (radians, ~2.87°)
- Info/MetaData/FL/Registration/Scale: scale correction factor (~0.887)
- Info/MetaData/FL/Registration/TranslationX: X offset in micrometers
- Info/MetaData/FL/Registration/TranslationY: Y offset in micrometers

## Resolution Info:
- Data/3D attrs: ResolutionX/Y/Z for HT (um/pixel)
- Data/3DFL attrs: ResolutionX/Y/Z for FL (um/pixel)  
- Data/3DFL/CH0 attrs: OffsetZ for FL Z-offset from HT origin (um)

## Z-axis Mapping:
FL typically images a subset of the HT Z range. The FL OffsetZ parameter
indicates where FL slice 0 starts in the HT Z coordinate system.
"""

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RegistrationParams:
    """FL to HT registration parameters."""
    # XY Registration
    rotation: float = 0.0        # radians
    scale: float = 1.0           # scale correction factor
    translation_x: float = 0.0   # micrometers
    translation_y: float = 0.0   # micrometers
    
    # Resolution (um/pixel)
    ht_res_x: float = 0.196
    ht_res_y: float = 0.196
    ht_res_z: float = 0.839
    fl_res_x: float = 0.122
    fl_res_y: float = 0.122
    fl_res_z: float = 1.044
    
    # Z offset
    fl_offset_z: float = 0.0     # micrometers from HT Z=0
    
    def __repr__(self):
        return (f"RegistrationParams(\n"
                f"  rotation={self.rotation:.6f} rad ({np.degrees(self.rotation):.2f}°),\n"
                f"  scale={self.scale:.6f},\n"
                f"  translation=({self.translation_x:.2f}, {self.translation_y:.2f}) um,\n"
                f"  ht_res=({self.ht_res_x:.4f}, {self.ht_res_y:.4f}, {self.ht_res_z:.4f}) um/px,\n"
                f"  fl_res=({self.fl_res_x:.4f}, {self.fl_res_y:.4f}, {self.fl_res_z:.4f}) um/px,\n"
                f"  fl_offset_z={self.fl_offset_z:.2f} um\n"
                f")")


def load_registration_params(tcf_file: h5py.File) -> RegistrationParams:
    """Load registration parameters from TCF file metadata."""
    params = RegistrationParams()
    
    # Load XY registration params
    if 'Info/MetaData/FL/Registration' in tcf_file:
        reg = tcf_file['Info/MetaData/FL/Registration']
        if 'Rotation' in reg.attrs:
            params.rotation = float(reg.attrs['Rotation'][0])
        if 'Scale' in reg.attrs:
            params.scale = float(reg.attrs['Scale'][0])
        if 'TranslationX' in reg.attrs:
            params.translation_x = float(reg.attrs['TranslationX'][0])
        if 'TranslationY' in reg.attrs:
            params.translation_y = float(reg.attrs['TranslationY'][0])
    
    # Load HT resolution
    if 'Data/3D' in tcf_file:
        ht_attrs = tcf_file['Data/3D'].attrs
        if 'ResolutionX' in ht_attrs:
            params.ht_res_x = float(ht_attrs['ResolutionX'][0])
        if 'ResolutionY' in ht_attrs:
            params.ht_res_y = float(ht_attrs['ResolutionY'][0])
        if 'ResolutionZ' in ht_attrs:
            params.ht_res_z = float(ht_attrs['ResolutionZ'][0])
    
    # Load FL resolution
    if 'Data/3DFL' in tcf_file:
        fl_attrs = tcf_file['Data/3DFL'].attrs
        if 'ResolutionX' in fl_attrs:
            params.fl_res_x = float(fl_attrs['ResolutionX'][0])
        if 'ResolutionY' in fl_attrs:
            params.fl_res_y = float(fl_attrs['ResolutionY'][0])
        if 'ResolutionZ' in fl_attrs:
            params.fl_res_z = float(fl_attrs['ResolutionZ'][0])
    
    # Load FL Z offset from channel
    if 'Data/3DFL/CH0' in tcf_file:
        ch_attrs = tcf_file['Data/3DFL/CH0'].attrs
        if 'OffsetZ' in ch_attrs:
            params.fl_offset_z = float(ch_attrs['OffsetZ'][0])
    
    return params


def register_fl_slice_to_ht(
    fl_slice: np.ndarray,
    ht_shape: Tuple[int, int],
    params: RegistrationParams = None
) -> np.ndarray:
    """
    Register a single FL slice to HT coordinate space.
    
    Both modalities cover the same physical FOV (~230 um), so registration
    is simply resampling FL to match HT pixel dimensions.
    
    Args:
        fl_slice: 2D FL image (Y, X)
        ht_shape: Target shape (HT_Y, HT_X)
        params: Registration parameters (unused, kept for API compatibility)
        
    Returns:
        Registered FL image with shape matching ht_shape
    """
    from scipy import ndimage
    
    ht_h, ht_w = ht_shape
    fl_h, fl_w = fl_slice.shape
    
    # Simple resize - both cover same physical FOV
    zoom_y = ht_h / fl_h
    zoom_x = ht_w / fl_w
    
    return ndimage.zoom(fl_slice.astype(float), (zoom_y, zoom_x), order=1)


def register_fl_volume_to_ht(
    fl_volume: np.ndarray,
    ht_shape: Tuple[int, int, int],
    params: RegistrationParams
) -> np.ndarray:
    """
    Register entire FL volume to HT coordinate space.
    
    Handles Z-axis mapping based on physical coordinates.
    
    Args:
        fl_volume: 3D FL data (Z, Y, X)
        ht_shape: Target shape (HT_Z, HT_Y, HT_X)
        params: Registration parameters (for Z-axis mapping)
        
    Returns:
        Registered FL volume in HT coordinate space
    """
    ht_z, ht_h, ht_w = ht_shape
    fl_z, fl_h, fl_w = fl_volume.shape
    
    # Initialize output
    output = np.zeros((ht_z, ht_h, ht_w), dtype=np.float32)
    
    # Map each HT Z slice to corresponding FL Z slice
    for ht_slice_idx in range(ht_z):
        # Physical Z position of this HT slice
        ht_z_um = ht_slice_idx * params.ht_res_z
        
        # Corresponding FL slice (accounting for Z offset)
        fl_z_um = ht_z_um - params.fl_offset_z
        fl_slice_idx = fl_z_um / params.fl_res_z
        
        # Check if within FL Z range
        if fl_slice_idx < 0 or fl_slice_idx >= fl_z - 1:
            continue
        
        # Interpolate between adjacent FL slices
        fl_z0 = int(np.floor(fl_slice_idx))
        fl_z1 = min(fl_z0 + 1, fl_z - 1)
        fz = fl_slice_idx - fl_z0
        
        # Interpolate FL slices
        fl_interp = (1 - fz) * fl_volume[fl_z0] + fz * fl_volume[fl_z1]
        
        # Register the interpolated slice (simple resize)
        output[ht_slice_idx] = register_fl_slice_to_ht(fl_interp, (ht_h, ht_w))
    
    return output


def compute_overlap_score(
    ht_image: np.ndarray,
    fl_registered: np.ndarray,
    ht_percentile: float = 75,
    fl_percentile: float = 85
) -> float:
    """
    Compute overlap score between HT and registered FL.
    
    Higher scores indicate better registration.
    """
    # Threshold HT
    ht_thresh = np.percentile(ht_image, ht_percentile)
    ht_mask = ht_image > ht_thresh
    
    # Threshold FL
    if np.max(fl_registered) == 0:
        return 0.0
    
    fl_valid = fl_registered[fl_registered > 0]
    if len(fl_valid) < 100:
        return 0.0
    
    fl_thresh = np.percentile(fl_valid, fl_percentile)
    fl_mask = fl_registered > fl_thresh
    
    # Score = fraction of FL bright pixels that overlap HT bright pixels
    if np.sum(fl_mask) > 0:
        return np.sum(fl_mask & ht_mask) / np.sum(fl_mask)
    return 0.0


def test_registration(tcf_path: str, output_dir: str = None):
    """
    Test registration on a TCF file and save visualization.
    """
    import matplotlib.pyplot as plt
    
    tcf_path = Path(tcf_path)
    if output_dir is None:
        output_dir = tcf_path.parent
    output_dir = Path(output_dir)
    
    print(f"Loading: {tcf_path.name}")
    
    with h5py.File(tcf_path, 'r') as f:
        # Check for FL data
        if 'Data/3DFL/CH0/000000' not in f:
            print("No FL data found in this file")
            return
        
        # Load data
        ht_3d = f['Data/3D/000000'][:]
        fl_3d = f['Data/3DFL/CH0/000000'][:]
        
        print(f"HT shape: {ht_3d.shape}")
        print(f"FL shape: {fl_3d.shape}")
        
        # Load params
        params = load_registration_params(f)
        print(f"\n{params}")
        
        # Use MIPs for visualization
        ht_mip = np.max(ht_3d, axis=0).astype(float)
        fl_mip = np.max(fl_3d, axis=0).astype(float)
        
        # Simple resize registration (both cover same FOV)
        fl_registered = register_fl_slice_to_ht(fl_mip, ht_mip.shape)
        
        # Calculate overlap score
        score = compute_overlap_score(ht_mip, fl_registered)
        print(f"\nOverlap score: {score*100:.1f}%")
        
        # Normalize for display
        def norm(img):
            p1, p99 = np.percentile(img[img > 0] if np.any(img > 0) else img, [1, 99])
            return np.clip((img - p1) / (p99 - p1 + 1e-10), 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(ht_mip, cmap='gray')
        axes[0, 0].set_title(f'HT MIP {ht_mip.shape}')
        
        axes[0, 1].imshow(fl_mip, cmap='Greens')
        axes[0, 1].set_title(f'FL MIP (raw) {fl_mip.shape}')
        
        axes[0, 2].imshow(fl_registered, cmap='Greens')
        axes[0, 2].set_title(f'FL registered {fl_registered.shape}')
        
        # Overlay
        ht_n = norm(ht_mip)
        fl_n = norm(fl_registered)
        
        rgb = np.zeros((*ht_mip.shape, 3))
        rgb[:, :, 0] = ht_n  # Red = HT
        rgb[:, :, 1] = fl_n  # Green = FL
        
        axes[1, 0].imshow(np.clip(rgb, 0, 1))
        axes[1, 0].set_title(f'Overlay (R=HT, G=FL)\nScore: {score*100:.1f}%')
        
        # Z-axis info
        ht_z_start = int(params.fl_offset_z / params.ht_res_z)
        ht_z_end = ht_z_start + int(fl_3d.shape[0] * params.fl_res_z / params.ht_res_z)
        axes[1, 1].text(0.5, 0.5, f'Z-axis mapping:\n\nFL offset: {params.fl_offset_z:.1f} um\n'
                        f'FL covers HT slices {ht_z_start}-{ht_z_end}\n'
                        f'(out of {ht_3d.shape[0]} total)',
                        ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Z-axis Info')
        axes[1, 1].axis('off')
        
        # Show a matched Z slice
        fl_z = fl_3d.shape[0] // 2
        ht_z = ht_z_start + int(fl_z * params.fl_res_z / params.ht_res_z)
        if 0 <= ht_z < ht_3d.shape[0]:
            ht_slice = ht_3d[ht_z].astype(float)
            fl_slice = fl_3d[fl_z].astype(float)
            fl_slice_reg = register_fl_slice_to_ht(fl_slice, ht_slice.shape)
            
            rgb2 = np.zeros((*ht_slice.shape, 3))
            rgb2[:, :, 0] = norm(ht_slice)
            rgb2[:, :, 1] = norm(fl_slice_reg)
            axes[1, 2].imshow(np.clip(rgb2, 0, 1))
            axes[1, 2].set_title(f'Single slice: HT z={ht_z}, FL z={fl_z}')
        else:
            axes[1, 2].axis('off')
        
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        output_path = output_dir / f'{tcf_path.stem}_registration.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved: {output_path}")
        plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        tcf_path = sys.argv[1]
    else:
        # Default test file
        tcf_path = r"c:\Dev\Repos\tomocube-tools\data\lab 2056\melanoma b16\260114.131315.melanoma b16.010.Group1.A1.S010\260114.131315.melanoma b16.010.Group1.A1.S010.TCF"
    
    test_registration(tcf_path, r"c:\Dev\Repos\tomocube-tools\output")
