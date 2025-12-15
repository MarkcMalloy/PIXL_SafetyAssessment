from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import pandas as pd

from src.config import Config  # same style as view_depth_calibrated.py


# ---------- helpers (copied/trimmed from view_depth_calibrated) ---------- #

def load_sli_points_for_overlay(csv_path: Path):
    """
    Load SLI CSV and return:
      u_px, v_px : pixel coords (0..W-1, 0..H-1)
      X_mm, Y_mm, Z_mm : camera frame in mm (with Y flipped)
    """
    df = pd.read_csv(csv_path)

    u_px = df["u_px"].to_numpy(dtype=float)
    v_px = df["v_px"].to_numpy(dtype=float)

    X_um = df["X_um"].to_numpy(dtype=float)
    Y_um = df["Y_um"].to_numpy(dtype=float)
    Z_um = df["Z_um"].to_numpy(dtype=float)

    X_mm = X_um / 1000.0
    Y_mm = Y_um / 1000.0
    Z_mm = Z_um / 1000.0

    # Flip Y to match image orientation (same convention as in view_depth_calibrated)
    Y_mm = -Y_mm

    return u_px, v_px, X_mm, Y_mm, Z_mm


def _make_grid_from_z(z: np.ndarray, pad_above: float = 0.5) -> pv.StructuredGrid:
    """
    Build a PyVista StructuredGrid from a Z map.

    pad_above: fraction of the z-range to shift the surface downward so there
               is more empty space above the terrain in the 3D view.
    """
    z = z.copy().astype(np.float32)
    valid = np.isfinite(z)
    if valid.any():
        z_min = float(z[valid].min())
        z_max = float(z[valid].max())
        z_range = z_max - z_min
        z[valid] -= pad_above * z_range  # shift down

    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]

    grid = pv.StructuredGrid()
    grid.points = np.c_[xx.ravel(order="C"),
                        yy.ravel(order="C"),
                        z.ravel(order="C")]
    grid.dimensions = (W, H, 1)
    grid["z"] = z.ravel(order="C")
    return grid


# ---------- main visualization ---------- #

def show_depth_map(
    depth_path: Path,
    label: str = "DEPTH",
    subtract_mean: bool = True,
    exaggeration: float = 1.0,
    texture_path: Path | None = None,
    sli_csv_path: Path | None = None,
):
    """
    Visualize a single depth map stored as depth.npy.

    Parameters
    ----------
    depth_path : Path
        Path to depth.npy.
    label : str
        Label for prints / window title.
    subtract_mean : bool
        If True, subtract mean depth for nicer visualization.
    exaggeration : float
        Extra vertical exaggeration factor for display only.
    texture_path : Path or None
        Optional image to drape as a texture on the terrain.
    sli_csv_path : Path or None
        Optional SLI CSV file to overlay SLI points.
    """
    depth_path = depth_path.expanduser().resolve()
    print(f"\n=== [{label}] view_depth ===")
    print(f"  Loading depth: {depth_path}")

    if not depth_path.is_file():
        raise FileNotFoundError(f"depth file not found: {depth_path}")

    z = np.load(depth_path).astype(np.float32)
    mask = np.isfinite(z)

    if not mask.any():
        print("  WARNING: no finite depth values to visualize.")
        return

    print(f"  depth range: {z[mask].min():.3f} .. {z[mask].max():.3f}")

    z_vis = z.copy()
    if subtract_mean:
        z_vis[mask] -= z_vis[mask].mean()

    z_vis[mask] *= exaggeration

    print(
        f"  z_vis range after scaling: "
        f"{z_vis[mask].min():.3f} .. {z_vis[mask].max():.3f}"
    )

    grid = _make_grid_from_z(z_vis)

    title = f"{label}: depth (mean-removed={subtract_mean}, exagg={exaggeration:.3f})"
    plotter = pv.Plotter(title=title, window_size=[1920, 1080])

    # Optional texture
    texture = None
    if texture_path is not None:
        texture_path = texture_path.expanduser()
        if texture_path.is_file():
            print(f"  Using texture: {texture_path}")
            texture = pv.read_texture(str(texture_path))
            grid.texture_map_to_plane(inplace=True)
        else:
            print(f"  WARNING: texture not found: {texture_path}")

    if texture is not None:
        plotter.add_mesh(grid, texture=texture, show_edges=False)
    else:
        plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)

    # Optional SLI overlay
    if sli_csv_path is not None:
        sli_csv_path = sli_csv_path.expanduser()
        if sli_csv_path.is_file():
            print(f"  Overlaying SLI points from: {sli_csv_path}")
            try:
                u_px, v_px, X_mm, Y_mm, Z_mm = load_sli_points_for_overlay(sli_csv_path)
            except Exception as e:
                print("  ERROR loading SLI CSV:", e)
            else:
                H, W = z.shape

                # Flip vertically only (mirror in Y) to match SfS/texture
                u_sfs = u_px
                v_sfs = (H - 1) - v_px

                # Sample SfS height at these locations
                u_idx = np.clip(np.rint(u_sfs).astype(int), 0, W - 1)
                v_idx = np.clip(np.rint(v_sfs).astype(int), 0, H - 1)
                Z_sfs = z_vis[v_idx, u_idx]

                pts = np.column_stack([u_sfs, v_sfs, Z_sfs])
                cloud = pv.PolyData(pts)
                cloud["Z_mm"] = Z_mm

                plotter.add_mesh(
                    cloud,
                    render_points_as_spheres=True,
                    point_size=13,
                    scalars="Z_mm",
                    cmap="viridis",
                    scalar_bar_args={"title": "SLI Z (mm)"},
                )
        else:
            print(f"  WARNING: SLI CSV not found: {sli_csv_path}")

    plotter.show_grid()
    plotter.show()


def main():
    texture = Path(Config.OUTPUT_DIR_COMPOSITES) / "composite_max.png"
    show_depth_map(
        depth_path=Path(Config.OUTPUT_DIR_DEPTH) / "depth.npy",
        label="Relative Depth",
        subtract_mean=True,
        exaggeration=5,
        texture_path=None,
        sli_csv_path=Path(Config.DEFAULT_SLI_CSV_GLOB_40mm)
    )
    show_depth_map(
        depth_path=Path(Config.OUTPUT_DIR_DEPTH) / "cal_depth.npy",
        label="Calibrated Depth",
        subtract_mean=True,
        exaggeration=5,
        texture_path=None,
        sli_csv_path=Path(Config.DEFAULT_SLI_CSV_GLOB_40mm)
    )


if __name__ == "__main__":
    main()
