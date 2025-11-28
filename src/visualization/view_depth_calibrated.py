from __future__ import annotations

from pathlib import Path
import numpy as np
import pyvista as pv

from src.config import Config  # adjust import if needed


# ---------- helpers ---------- #

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
        exaggeration: float = 1.0,
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

    plotter = pv.Plotter(title=title,window_size=[1920, 1920])
    plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    plotter.show_grid()
    plotter.show()


# ---------- run for both datasets ---------- #

def run_for_dataset(base_dir: Path, label: str, a: float | None = None):
    """
    Run all three viewers for a single dataset.

    Parameters
    ----------
    base_dir : Path
        Output folder for this dataset.
    label : str
        Label for titles.
    a : float or None
        Calibration slope used in main.py for this dataset; copy from:
          "[LABEL] SfS–SLI calibration: scale a = ..."
        If None, we won't rescale by 'a' in the plots.
    """
    compare_p_q(base_dir, label)
    compare_relative_calibrated(base_dir, label, a=a)
    show_sfs_calibrated(base_dir, label, a=a,
                        subtract_mean=True, exaggeration=1.0)


def main():
    # Use the output dirs defined in Config
    base_no_obstacle = Path(Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE)
    base_obstacle = Path(Config.OUTPUT_DIR_DEPTH_OBSTACLE)

    print(f"NO_OBSTACLE output dir: {base_no_obstacle}")
    print(f"OBSTACLE  output dir: {base_obstacle}")

    # === IMPORTANT ===
    # Put in the calibration slopes 'a' from your console output
    # for each dataset, or set to None to skip a-based rescaling.
    a_no_obstacle = 0.179953  # e.g. 0.179953 for rho = 1, 0.486064 for rho = 0.55
    a_obstacle    = 0.5   # e.g. -0.235702 for rho = 1, -0.220645 for rho = 0.55

    # NO_OBSTACLE dataset
    #run_for_dataset(base_no_obstacle, label="NO_OBSTACLE", a=a_no_obstacle)

    # OBSTACLE dataset
    run_for_dataset(base_obstacle, label="OBSTACLE", a=a_obstacle)


if __name__ == "__main__":
    main()
