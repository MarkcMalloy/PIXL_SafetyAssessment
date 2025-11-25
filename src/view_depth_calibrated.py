from pathlib import Path
import numpy as np
import pyvista as pv


def main():
    # Paths
    z_path = Path("../Output/Depth/depth_calibrated.npy")
    zc_path = Path("../Output/Depth/depth_calibrated.npy")

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


if __name__ == "__main__":
    main()
