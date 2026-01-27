"""
TCF Types - Data classes for TCF file handling.

This module contains the core data structures:
    - RegistrationParams: FL-to-HT registration parameters
    - ViewerState: Mutable state container for viewers
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tomocube.core.constants import (
    DEFAULT_HT_RES_X,
    DEFAULT_HT_RES_Y,
    DEFAULT_HT_RES_Z,
    DEFAULT_FL_RES_X,
    DEFAULT_FL_RES_Y,
    DEFAULT_FL_RES_Z,
)


@dataclass
class RegistrationParams:
    """FL to HT registration parameters from TCF metadata."""

    # XY Registration
    rotation: float = 0.0        # radians
    scale: float = 1.0           # scale correction factor
    translation_x: float = 0.0   # micrometers
    translation_y: float = 0.0   # micrometers

    # Resolution (um/pixel)
    ht_res_x: float = DEFAULT_HT_RES_X
    ht_res_y: float = DEFAULT_HT_RES_Y
    ht_res_z: float = DEFAULT_HT_RES_Z
    fl_res_x: float = DEFAULT_FL_RES_X
    fl_res_y: float = DEFAULT_FL_RES_Y
    fl_res_z: float = DEFAULT_FL_RES_Z

    # Z offset (default for channels without specific offset)
    fl_offset_z: float = 0.0     # micrometers from HT Z=0

    # Per-channel Z offsets (channel name -> offset in micrometers)
    # If a channel is not in this dict, fl_offset_z is used as fallback
    channel_offsets_z: dict[str, float] | None = None

    def get_offset_z(self, channel: str | None = None) -> float:
        """Get Z offset for a specific channel, or default if not specified."""
        if channel and self.channel_offsets_z and channel in self.channel_offsets_z:
            return self.channel_offsets_z[channel]
        return self.fl_offset_z

    def __repr__(self) -> str:
        offsets_str = ""
        if self.channel_offsets_z:
            offsets_str = f",\n  channel_offsets_z={self.channel_offsets_z}"
        return (
            f"RegistrationParams(\n"
            f"  rotation={self.rotation:.4f} rad ({np.degrees(self.rotation):.2f}°),\n"
            f"  scale={self.scale:.4f},\n"
            f"  translation=({self.translation_x:.2f}, {self.translation_y:.2f}) µm,\n"
            f"  ht_res=({self.ht_res_x:.4f}, {self.ht_res_y:.4f}, {self.ht_res_z:.4f}) µm/px,\n"
            f"  fl_res=({self.fl_res_x:.4f}, {self.fl_res_y:.4f}, {self.fl_res_z:.4f}) µm/px,\n"
            f"  fl_offset_z={self.fl_offset_z:.2f} µm{offsets_str}\n"
            f")"
        )


@dataclass
class ViewerState:
    """
    Container for viewer state, separating state from behavior.
    
    This dataclass holds all mutable state for the viewer, making it easy
    to track what data is being modified and enabling easier testing.
    """
    
    # Navigation
    current_z: int = 0
    current_y: int = 0
    current_x: int = 0
    current_timepoint: int = 0
    
    # Display
    colormap: str = "gray"
    invert_cmap: bool = False
    vmin: float = 0.0
    vmax: float = 1.0
    
    # Fluorescence
    show_fluorescence: bool = False
    fl_overlay_alpha: float = 0.5
    current_fl_channel: str | None = None
    fl_vmin: float = 0.0
    fl_vmax: float = 1.0
    
    def get_cmap(self) -> str:
        """Get current colormap name with optional inversion."""
        return f"{self.colormap}_r" if self.invert_cmap else self.colormap
