from pathlib import Path
import numpy as np
import pyvista as pv

def compare_relative_calibrated():
    # Paths
    z_path = Path("../Output/sfs_depth_relative.npy")  # relative SfS
    zc_path = Path("../Output/sfs_depth_calibrated_mm.npy")  # calibrated

    print(f"Loading {z_path}")
    print(f"Loading {zc_path}")

    # Load arrays
    z = np.load(z_path)
    zc = np.load(zc_path)

    print("z range:", z.min(), z.max())
    print("zc range:", zc.min(), zc.max())

    # -----------------------------
    # Correlation (mask-aware)
    # -----------------------------
    mask = np.isfinite(zc) & np.isfinite(z)

    z_norm = z[mask] - z[mask].mean()
    zc_norm = zc[mask] - zc[mask].mean()

    corr = np.corrcoef(z_norm.flatten(), zc_norm.flatten())[0, 1]
    print("corr:", corr)

    # -----------------------------
    # Visualization for calibrated depth
    # -----------------------------
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # Mean-removal for visualization
    zc_vis = zc.copy()
    valid = np.isfinite(zc_vis)

    mean_zc = zc_vis[valid].mean()
    zc_vis[valid] -= mean_zc

    # -------------------------------------------------------------
    # RESCALE BY |a| TO RESTORE RELIEF VISIBILITY
    # -------------------------------------------------------------
    # Put your printed calibration slope here:
    a = -0.4327   # <-- from console output

    if abs(a) > 1e-6:
        zc_vis[valid] /= abs(a)

    print(f"zc_vis range: {zc_vis[valid].min():.3f} .. {zc_vis[valid].max():.3f}")

    # -----------------------------
    # Build PyVista structured grid
    # -----------------------------
    grid = pv.StructuredGrid()
    grid.points = np.c_[xx.ravel(order="C"),
                        yy.ravel(order="C"),
                        zc_vis.ravel(order="C")]
    grid.dimensions = (W, H, 1)
    grid["z"] = zc_vis.ravel(order="C")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    plotter.show_grid()
    plotter.show()

def compare_p_q():
    # Paths
    z_path = Path("../Output/Depth/NoObstacle/sfs_p.npy")  # relative SfS
    zc_path = Path("../Output/Depth/NoObstacle/sfs_q.npy")  # calibrated

    print(f"Loading {z_path}")
    print(f"Loading {zc_path}")

    # Load arrays
    z = np.load(z_path)
    zc = np.load(zc_path)

    print("z range:", z.min(), z.max())
    print("zc range:", zc.min(), zc.max())

    # -----------------------------
    # Correlation (mask-aware)
    # -----------------------------
    mask = np.isfinite(zc) & np.isfinite(z)

    z_norm = z[mask] - z[mask].mean()
    zc_norm = zc[mask] - zc[mask].mean()

    corr = np.corrcoef(z_norm.flatten(), zc_norm.flatten())[0, 1]
    print("corr:", corr)

    # -----------------------------
    # Visualization for calibrated depth
    # -----------------------------
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # Mean-removal for visualization
    zc_vis = zc.copy()
    valid = np.isfinite(zc_vis)

    mean_zc = zc_vis[valid].mean()
    zc_vis[valid] -= mean_zc

    # -------------------------------------------------------------
    # RESCALE BY |a| TO RESTORE RELIEF VISIBILITY
    # -------------------------------------------------------------
    # Put your printed calibration slope here:
    a = -0.4327   # <-- from console output

    if abs(a) > 1e-6:
        zc_vis[valid] /= abs(a)

    print(f"zc_vis range: {zc_vis[valid].min():.3f} .. {zc_vis[valid].max():.3f}")

    # -----------------------------
    # Build PyVista structured grid
    # -----------------------------
    grid = pv.StructuredGrid()
    grid.points = np.c_[xx.ravel(order="C"),
                        yy.ravel(order="C"),
                        zc_vis.ravel(order="C")]
    grid.dimensions = (W, H, 1)
    grid["z"] = zc_vis.ravel(order="C")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    plotter.show_grid()
    plotter.show()

def show_sfs(
        a=None,                # calibration slope from main.py, e.g. -0.235702
        exaggeration=1.0,      # extra vertical exaggeration for viewing
        subtract_mean=True,    # remove mean to center the surface
        calibrated=True
):
    """
    Visualize the calibrated SfS depth (sfs_depth_calibrated_mm.npy).

    Parameters
    ----------
    a : float or None
        Calibration slope printed by main.py:
            SfS–SLI calibration: scale a = ..., offset b = ...
        If provided, we divide by -a to undo the compression AND flip
        so that the shape matches the relative depth visually.
        If None, we just show the calibrated depth in mm (up to a
        global offset), without “undoing” anything.

    exaggeration : float
        Extra vertical scale factor applied after the above steps.

    subtract_mean : bool
        If True, subtract the mean depth before plotting so the relief
        is centered around zero (helps visualization).
    """
    if calibrated:
        path = Path("../Output/Depth/NoObstacle/sfs_depth_calibrated_mm.npy")
    else:
        path = Path("../Output/Depth/NoObstacle/sfs_depth_relative.npy")
    zc_path = path
    print(f"Loading calibrated depth from {zc_path}")
    zc = np.load(zc_path).astype(np.float32)

    print("zc range (mm):", float(zc.min()), "to", float(zc.max()))

    H, W = zc.shape
    yy, xx = np.mgrid[0:H, 0:W]

    zc_vis = zc.copy()
    mask = np.isfinite(zc_vis)

    if subtract_mean:
        mean_zc = zc_vis[mask].mean()
        zc_vis[mask] -= mean_zc

    # Optionally undo calibration scale & flip sign so bulge is up
    if a is not None and abs(a) > 1e-6:
        # divide by -a: undo compression (|a| < 1) and flip sign (a < 0)
        zc_vis[mask] /= -a

    # Extra vertical exaggeration for display only
    zc_vis[mask] *= exaggeration

    print(
        f"zc_vis range after scaling: "
        f"{float(zc_vis[mask].min()):.3f} .. {float(zc_vis[mask].max()):.3f}"
    )

    # Build PyVista structured grid
    grid = pv.StructuredGrid()
    grid.points = np.c_[xx.ravel(order="C"),
                        yy.ravel(order="C"),
                        zc_vis.ravel(order="C")]
    grid.dimensions = (W, H, 1)
    grid["z"] = zc_vis.ravel(order="C")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="z", cmap="viridis", show_edges=False)
    plotter.show_grid()
    plotter.show()

def main():
    compare_p_q()
    compare_relative_calibrated()
    # Use the current 'a' from your console, e.g. -0.235702 if showing calibrated depth. Else use 0
    #show_sfs(a=0.235702, exaggeration=1, calibrated=True) # WithObstacle
    show_sfs(a=0.179953, exaggeration=1, calibrated=True) # NoObstacle
    #show_sfs(a=0.235702, exaggeration=1.5, calibrated=True)
    #show_sfs(a=0.235702, exaggeration=1.75, calibrated=True)
    #show_sfs(a=0.235702, exaggeration=1, calibrated=True)




if __name__ == "__main__":
    main()
