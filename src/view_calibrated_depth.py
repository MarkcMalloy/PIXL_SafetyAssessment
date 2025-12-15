#!/usr/bin/env python3
"""
view_calibrated_depth.py

Visualize a calibrated depth map (cal_depth.npy) as a 3D mesh
with physically meaningful scaling in all axes (X, Y, Z in mm).

Requirements:
    pip install numpy pyvista

Usage:
    python view_calibrated_depth.py \
        --depth /path/to/cal_depth.npy \
        --pixel-size-mm 0.02

Pixel size:
    pixel-size-mm is the physical size of a pixel on the sample
    in millimetres. Adjust to your calibration (e.g. 0.01, 0.02, etc).
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from src.config import Config  # <- change to `from src import Config` if needed


def load_depth(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {arr.shape}")
    return arr.astype(np.float32)


def build_surface_from_depth(
    depth_mm: np.ndarray,
    pixel_size_mm: float,
) -> pv.StructuredGrid:
    """
    Create a PyVista StructuredGrid from a depth map in mm.

    X, Y, Z are all in millimetres.
    """
    nrows, ncols = depth_mm.shape  # (H, W)

    # Coordinate grids in mm
    x = np.arange(ncols, dtype=np.float32) * pixel_size_mm
    y = np.arange(nrows, dtype=np.float32) * pixel_size_mm
    xx, yy = np.meshgrid(x, y)

    # Ensure C-contiguous arrays for PyVista
    xx = np.ascontiguousarray(xx)
    yy = np.ascontiguousarray(yy)
    zz = np.ascontiguousarray(depth_mm)

    # Each point (xx, yy, zz) is in mm → physically correct scaling
    grid = pv.StructuredGrid(xx, yy, zz)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth",
        type=str,
        default=str(Path(Config.OUTPUT_DIR_DEPTH) / "cal_depth.npy"),
        help="Path to calibrated depth map (cal_depth.npy, in mm).",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.02,
        help="Physical pixel size on sample in millimetres.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional boolean mask .npy; pixels with mask==False are hidden.",
    )
    parser.add_argument(
        "--show-rel",
        type=str,
        default=None,
        help="Optional relative depth map (depth.npy) for debugging stats.",
    )
    parser.add_argument(
        "--center-z",
        action="store_true",
        help=(
            "If set, subtract mean Z so the surface is centred around 0 mm. "
            "This only affects visualization, not the saved file."
        ),
    )
    args = parser.parse_args()

    depth_path = Path(args.depth)
    if not depth_path.exists():
        raise SystemExit(f"Depth file not found: {depth_path}")

    print(f"[VIEW] Loading calibrated depth from {depth_path}")
    z_cal = load_depth(depth_path)  # mm

    # Optional: load relative depth just for stats/debug
    if args.show_rel is not None:
        rel_path = Path(args.show_rel)
        if rel_path.exists():
            z_rel = load_depth(rel_path)
            print(
                f"[STATS] z_rel range: {np.nanmin(z_rel):.3f} .. "
                f"{np.nanmax(z_rel):.3f}"
            )

    # Optional mask: hide invalid/background pixels
    mask = None
    if args.mask is not None:
        mask_path = Path(args.mask)
        if mask_path.exists():
            print(f"[VIEW] Using mask: {mask_path}")
            mask = np.load(mask_path).astype(bool)
            if mask.shape != z_cal.shape:
                raise SystemExit(
                    f"Mask shape {mask.shape} does not match depth shape {z_cal.shape}"
                )
        else:
            print(f"[VIEW] Mask not found, ignoring: {mask_path}")

    # Apply mask: for visualization, set masked pixels to NaN
    if mask is not None:
        z_vis = z_cal.copy()
        z_vis[~mask] = np.nan
    else:
        z_vis = z_cal

    # Optionally centre Z around 0 (just for nicer viewing)
    if args.center_z:
        mean_z = np.nanmean(z_vis)
        print(f"[VIEW] Centering Z around 0 (subtracting mean {mean_z:.3f} mm)")
        z_vis = z_vis - mean_z

    print(
        f"[STATS] cal_depth range (mm, after centering={args.center_z}): "
        f"{np.nanmin(z_vis):.3f} .. {np.nanmax(z_vis):.3f}"
    )

    # Build surface (X, Y, Z all in mm → physically correct aspect ratio)
    surface = build_surface_from_depth(
        depth_mm=z_vis,
        pixel_size_mm=args.pixel_size_mm,
    )

    # --- Plot 3D mesh only ---
    p = pv.Plotter()
    p.enable_anti_aliasing()

    p.add_text("Calibrated 3D surface (mm)", font_size=10)
    p.add_mesh(
        surface,
        show_edges=False,
        nan_opacity=0.0,
        scalar_bar_args=dict(
            title="Z (mm)",
            vertical=True,
        ),
    )

    p.show_grid()
    p.show_axes()

    # Ensure we use real-world scaling: no extra exaggeration
    p.set_scale(1.0, 1.0, 0.3)

    # Nice isometric view
    p.camera_position = "iso"
    p.camera.zoom(1.2)

    p.show(
        title="Calibrated depth mesh (physical scale in mm)",
        window_size=(1600, 1000),
    )


if __name__ == "__main__":
    main()
