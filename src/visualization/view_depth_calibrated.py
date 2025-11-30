from __future__ import annotations

from pathlib import Path
import numpy as np
import pyvista as pv
import pandas as pd

from src.config import Config  # adjust import if needed


# ---------- helpers ---------- #
def load_sli_points_for_overlay(csv_path: Path):
    """
    Load SLI CSV the same way as visualize_sli_vectors.load_sli_points:

      u_px, v_px : pixel coords (0..W-1, 0..H-1)
      X_mm, Y_mm, Z_mm : camera frame in mm (with Y flipped)

    We only use u_px, v_px, Z_mm for overlay.
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

    # Flip Y to match image orientation (same as visualize_sli_vectors)
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
        # shift down by pad_above * range
        z[valid] -= pad_above * z_range

    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]

    grid = pv.StructuredGrid()
    grid.points = np.c_[xx.ravel(order="C"),
                        yy.ravel(order="C"),
                        z.ravel(order="C")]
    grid.dimensions = (W, H, 1)
    grid["z"] = z.ravel(order="C")
    return grid



def _load_depths(base_dir: Path):
    """Load relative and calibrated depth from a dataset output folder."""
    z_rel_path = base_dir / "sfs_depth_relative.npy"
    z_cal_path = base_dir / "sfs_depth_calibrated_mm.npy"

    print(f"  Loading relative depth:   {z_rel_path}")
    print(f"  Loading calibrated depth: {z_cal_path}")

    z_rel = np.load(z_rel_path)
    z_cal = np.load(z_cal_path)
    return z_rel, z_cal


def _load_pq(base_dir: Path):
    """Load slope fields p, q from a dataset output folder."""
    p_path = base_dir / "sfs_p.npy"
    q_path = base_dir / "sfs_q.npy"

    print(f"  Loading p: {p_path}")
    print(f"  Loading q: {q_path}")

    p = np.load(p_path)
    q = np.load(q_path)
    return p, q


# ---------- 1) compare relative vs calibrated ---------- #

def compare_relative_calibrated(base_dir: Path, label: str, a: float | None = None):
    """
    Compare relative and calibrated depth for one dataset:
      - print basic stats & correlation,
      - show both as separate surfaces.

    If 'a' is provided, we also show an alternative calibrated view with
    depth divided by -a (undo compression and flip sign) to visually match
    the relative relief.
    """
    print(f"\n=== [{label}] Compare relative vs calibrated ===")
    z_rel, z_cal = _load_depths(base_dir)

    mask = np.isfinite(z_rel) & np.isfinite(z_cal)
    if not mask.any():
        print("  WARNING: no valid overlapping pixels.")
        return

    zv = z_rel[mask].astype(np.float32)
    zcv = z_cal[mask].astype(np.float32)

    corr = float(np.corrcoef(zv - zv.mean(), zcv - zcv.mean())[0, 1])
    print(f"  z_rel range: {zv.min():.3f} .. {zv.max():.3f}")
    print(f"  z_cal range: {zcv.min():.3f} .. {zcv.max():.3f}")
    print(f"  correlation: {corr:.4f}")

    # Mean-remove for nicer visualization
    z_rel_vis = z_rel.copy()
    z_cal_vis = z_cal.copy()
    z_rel_vis[mask] -= z_rel_vis[mask].mean()
    z_cal_vis[mask] -= z_cal_vis[mask].mean()

    # Optionally, prepare a second calibrated view with |a|-rescaling
    if a is not None and abs(a) > 1e-6:
        z_cal_rescaled = z_cal_vis.copy()
        # divide by -a: undo compression and flip so bulge is up if a<0
        z_cal_rescaled[mask] /= -a
    else:
        z_cal_rescaled = z_cal_vis

    grid_rel = _make_grid_from_z(z_rel_vis)
    grid_cal = _make_grid_from_z(z_cal_rescaled)

    plotter = pv.Plotter(shape=(1, 2), title=f"{label}: relative vs calibrated",window_size=[1920, 1080])
    plotter.subplot(0, 0)
    plotter.add_mesh(grid_rel, scalars="z", cmap="viridis", show_edges=False)
    plotter.add_text("Relative depth (mean-removed)", font_size=10)
    plotter.show_grid()

    plotter.subplot(0, 1)
    if a is None:
        txt = "Calibrated depth (mm, mean-removed)"
    else:
        txt = f"Calibrated (mean-removed, /-a, a={a:.4f})"
    plotter.add_mesh(grid_cal, scalars="z", cmap="viridis", show_edges=False)
    plotter.add_text(txt, font_size=10)
    plotter.show_grid()

    plotter.link_views()
    plotter.show()


# ---------- 2) compare p & q ---------- #

def compare_p_q(base_dir: Path, label: str):
    """
    Visualize p and q slope fields for one dataset as surfaces.
    """
    print(f"\n=== [{label}] Compare p and q ===")
    p, q = _load_pq(base_dir)

    print(f"  p range: {p.min():.3f} .. {p.max():.3f}")
    print(f"  q range: {q.min():.3f} .. {q.max():.3f}")

    grid_p = _make_grid_from_z(p)
    grid_q = _make_grid_from_z(q)

    plotter = pv.Plotter(shape=(1, 2), title=f"{label}: p and q slopes",window_size=[1920, 1080])
    plotter.subplot(0, 0)
    plotter.add_mesh(grid_p, scalars="z", cmap="coolwarm", show_edges=False)
    plotter.add_text("p = dz/dx", font_size=10)
    plotter.show_grid()

    plotter.subplot(0, 1)
    plotter.add_mesh(grid_q, scalars="z", cmap="coolwarm", show_edges=False)
    plotter.add_text("q = dz/dy", font_size=10)
    plotter.show_grid()

    plotter.link_views()
    plotter.show()


# ---------- 3) show calibrated SfS terrain map ---------- #

def show_sfs_calibrated(
        base_dir: Path,
        label: str,
        a: float | None = None,
        subtract_mean: bool = True,
        exaggeration: float = 0.1,
        texture_path: Path | None = None,
        sli_csv_path: Path | None = None,     # <--- NEW
):
    """
    Show only the calibrated SfS depth as a terrain map for one dataset.

    Parameters
    ----------
    base_dir : Path
        Folder containing sfs_depth_calibrated_mm.npy
    label : str
        Label for prints / window titles.
    a : float or None
        Calibration slope from main.py ("scale a = ...").
        - If None: show true calibrated depth in mm (up to global mean).
        - If set: divide by -a to undo compression and flip sign for
          visual comparison with the relative map.
    subtract_mean : bool
        If True, subtract mean depth for nicer visualization.
    exaggeration : float
        Extra vertical exaggeration factor for display only.
    """
    print(f"\n=== [{label}] Show calibrated SfS terrain ===")
    z_cal_path = base_dir / "sfs_depth_calibrated_mm.npy"
    print(f"  Loading calibrated depth: {z_cal_path}")

    zc = np.load(z_cal_path).astype(np.float32)
    mask = np.isfinite(zc)

    if not mask.any():
        print("  WARNING: no finite depth values to visualize.")
        return

    print(f"  z_cal range (mm): {zc[mask].min():.3f} .. {zc[mask].max():.3f}")

    zc_vis = zc.copy()
    if subtract_mean:
        zc_vis[mask] -= zc_vis[mask].mean()

    # Optional: undo scale & flip for visual comparison
    if a is not None and abs(a) > 1e-6:
        zc_vis[mask] /= -a

    zc_vis[mask] *= exaggeration

    print(
        f"  zc_vis range after scaling: "
        f"{zc_vis[mask].min():.3f} .. {zc_vis[mask].max():.3f}"
    )

    grid = _make_grid_from_z(zc_vis)

    if a is None:
        title = f"{label}: calibrated SfS (mm, mean-removed)"
    else:
        title = f"{label}: calibrated SfS (mean-removed, /-a, a={a:.4f})"

    plotter = pv.Plotter(title=title, window_size=[1920, 1080])

    texture = None
    if texture_path is not None and texture_path.is_file():
        # Load the composite image as a PyVista texture
        texture = pv.read_texture(str(texture_path))
        # Map (x,y) → texture coords in [0,1] using the grid bounds
        grid.texture_map_to_plane(inplace=True)

    if texture is not None:
        # Show geometry with the composite image draped on top
        plotter.add_mesh(grid, texture=texture, show_edges=False)
    else:
        # Fallback: plain height-colored surface
        plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    # ------------------------------
    # Add SLI points if available
    # ------------------------------
    if sli_csv_path is not None and sli_csv_path.is_file():
        try:
            u_px, v_px, X_mm, Y_mm, Z_mm = load_sli_points_for_overlay(sli_csv_path)
        except Exception as e:
            print("ERROR loading SLI CSV:", e)
        else:
            H, W = zc.shape

            # ---- Flip vertically only (mirror in Y) ----
            # Keep u as-is; flip v so SLI uses same origin as SfS+texture.
            u_sfs = u_px
            v_sfs = (H - 1) - v_px

            # Sample SfS height at these locations
            u_idx = np.clip(np.rint(u_sfs).astype(int), 0, W - 1)
            v_idx = np.clip(np.rint(v_sfs).astype(int), 0, H - 1)
            Z_sfs = zc_vis[v_idx, u_idx]

            # Pixel-frame coordinates after flip
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
    plotter.show_grid()
    plotter.show()




# ---------- run for both datasets ---------- #

def run_for_dataset(base_dir, label, a=None, texture_path=None, sli_csv_path=None):
    compare_p_q(base_dir, label)
    compare_relative_calibrated(base_dir, label, a=a)
    show_sfs_calibrated(
        base_dir, label,
        a=a,
        subtract_mean=True,
        exaggeration=1.0,
        texture_path=texture_path,
        sli_csv_path=sli_csv_path,   # <--- PASS IT
    )


def main():
    # Use the output dirs defined in Config
    base_no_obstacle = Path(Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE)
    base_obstacle    = Path(Config.OUTPUT_DIR_DEPTH_OBSTACLE)

    print(f"NO_OBSTACLE output dir: {base_no_obstacle}")
    print(f"OBSTACLE  output dir: {base_obstacle}")

    a_no_obstacle = -0.355315
    a_obstacle    = 0.136831

    tex_no  = base_no_obstacle / "sfs_input_composite.png"
    tex_obs = base_obstacle    / "sfs_input_composite.png"

    # NO_OBSTACLE dataset
    #run_for_dataset(base_no_obstacle, label="NO_OBSTACLE",a=a_no_obstacle, texture_path=tex_no)

    # OBSTACLE dataset
    run_for_dataset(base_obstacle, label="OBSTACLE", a=a_obstacle, texture_path=tex_obs,sli_csv_path=Path(Config.SLI_CSV_GLOB_OBSTACLE))



if __name__ == "__main__":
    main()
