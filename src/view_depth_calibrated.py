from pathlib import Path
import numpy as np
import pyvista as pv
from config import Config
# ---------- small helpers ---------- #

def _make_grid_from_z(z: np.ndarray) -> pv.StructuredGrid:
    """Build a PyVista StructuredGrid from a Z map."""
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

def compare_relative_calibrated(base_dir: Path, label: str):
    """
    Compare relative and calibrated depth for one dataset:
      - print basic stats & correlation,
      - show both as separate surfaces.
    """
    print(f"\n=== [{label}] Compare relative vs calibrated ===")
    z_rel, z_cal = _load_depths(base_dir)

    mask = np.isfinite(z_rel) & np.isfinite(z_cal)
    if not mask.any():
        print("  WARNING: no valid overlapping pixels.")
        return

    zv = z_rel[mask].astype(np.float32)
    zcv = z_cal[mask].astype(np.float32)

    corr = float(np.corrcoef(zv, zcv)[0, 1])
    print(f"  z_rel range: {zv.min():.3f} .. {zv.max():.3f}")
    print(f"  z_cal range: {zcv.min():.3f} .. {zcv.max():.3f}")
    print(f"  correlation: {corr:.4f}")

    # Mean-remove for nicer visualization
    z_rel_vis = z_rel.copy()
    z_cal_vis = z_cal.copy()
    z_rel_vis[mask] -= z_rel_vis[mask].mean()
    z_cal_vis[mask] -= z_cal_vis[mask].mean()

    grid_rel = _make_grid_from_z(z_rel_vis)
    grid_cal = _make_grid_from_z(z_cal_vis)

    plotter = pv.Plotter(shape=(1, 2), title=f"{label}: relative vs calibrated")
    plotter.subplot(0, 0)
    plotter.add_mesh(grid_rel, scalars="z", cmap="viridis", show_edges=False)
    plotter.add_text("Relative depth", font_size=10)
    plotter.show_grid()

    plotter.subplot(0, 1)
    plotter.add_mesh(grid_cal, scalars="z", cmap="viridis", show_edges=False)
    plotter.add_text("Calibrated depth (mm, mean-removed)", font_size=10)
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

    plotter = pv.Plotter(shape=(1, 2), title=f"{label}: p and q slopes")
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
        A label for window titles / prints.
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

    zc_vis[mask] *= exaggeration

    print(
        f"  zc_vis range after mean/exaggeration: "
        f"{zc_vis[mask].min():.3f} .. {zc_vis[mask].max():.3f}"
    )

    grid = _make_grid_from_z(zc_vis)

    plotter = pv.Plotter(title=f"{label}: calibrated SfS (mm)")
    plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    plotter.show_grid()
    plotter.show()


# ---------- run for both datasets ---------- #

def run_for_dataset(base_dir: Path, label: str):
    """Run all three viewers for a single dataset."""
    compare_p_q(base_dir, label)
    compare_relative_calibrated(base_dir, label)
    show_sfs_calibrated(base_dir, label, subtract_mean=True, exaggeration=1.0)


def main():
    # Use the output dirs defined in Config
    base_no_obstacle = Path(Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE)
    base_obstacle = Path(Config.OUTPUT_DIR_DEPTH_OBSTACLE)

    print(f"NO_OBSTACLE output dir: {base_no_obstacle}")
    print(f"OBSTACLE  output dir: {base_obstacle}")

    # Run for NO_OBSTACLE dataset
    run_for_dataset(base_no_obstacle, label="NO_OBSTACLE")

    # Run for OBSTACLE dataset
    run_for_dataset(base_obstacle, label="OBSTACLE")


if __name__ == "__main__":
    main()
