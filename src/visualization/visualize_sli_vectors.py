from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from src.config import Config  # keep as you had it


def load_sli_points(csv_path):
    """
    Load SLI CSV and return:
      - u_px, v_px (pixel coords)
      - X_mm, Y_mm, Z_mm (camera frame in mm)
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

    # Flip camera Y to match image orientation
    Y_mm = -Y_mm

    return u_px, v_px, X_mm, Y_mm, Z_mm

def export_calibrated_sli_csv(
    csv_path: str,
    output_name: str,
    tilt_deg: float = 0.0,
    calibrateTilt: bool = False,
):
    """
    Load an SLI CSV (u_px, v_px, X_um, Y_um, Z_um),
    optionally remove a global tilt in Z as a function of Y,
    and save a calibrated CSV with the same structure in the
    directory of visualize_sli_vectors.py.

    Tilt correction is done in the *raw camera frame*:

        Z_cal = Z - tan(tilt_deg) * (Y_mm - Y0)

    where Y0 is the mean of Y_mm, so the average depth is preserved.
    """
    # Load original CSV
    df = pd.read_csv(csv_path)

    # Basic sanity: expect these columns
    required_cols = ["u_px", "v_px", "X_um", "Y_um", "Z_um"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    if calibrateTilt and tilt_deg != 0.0:
        # Convert to mm for the math
        Y_mm = df["Y_um"].astype(float).to_numpy() / 1000.0
        Z_mm = df["Z_um"].astype(float).to_numpy() / 1000.0

        theta = np.deg2rad(tilt_deg)
        m = np.tan(theta)              # dZ/dY slope
        Y0 = float(np.mean(Y_mm))      # keep mean depth level

        Z_mm_cal = Z_mm - m * (Y_mm - Y0)

        # Back to micrometers for storage
        df["Z_um"] = Z_mm_cal * 1000.0

    # Figure out where visualize_sli_vectors.py lives
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / output_name

    df.to_csv(out_path, index=False)
    print(f"Saved calibrated SLI CSV to: {out_path}")


def plot_sli_points(
    csv_path,
    img_width_px=752,
    img_height_px=580,
    image_path=None,
    tilt_deg=0.0,         # tilt angle IN DEGREES
    calibrateTilt=False,  # apply tilt correction or not
):
    # -------- Load raw SLI points (camera coordinates) --------
    u_px, v_px, x_mm, y_mm, z_mm = load_sli_points(csv_path)

    # -------- Apply tilt correction in RAW CAMERA FRAME --------
    if calibrateTilt and tilt_deg != 0.0:
        theta = np.deg2rad(tilt_deg)
        m = np.tan(theta)          # slope dZ/dY in camera frame
        Y0 = float(np.mean(y_mm))  # keep mean depth level the same
        z_mm = z_mm - m * (y_mm - Y0)

    # How much to exaggerate depth in the pixel view
    Z_SCALE_LEFT = 5.0

    # Flip Z for visualization so closest depth is highest
    Z_disp = -z_mm  # used for geometry; Z_mm still used for colouring

    # ---- Pixel view: use scaled Z for display ----
    z_center = float(np.mean(Z_disp))
    Z_disp_left = (Z_disp - z_center) * Z_SCALE_LEFT + z_center

    # Z-limits for the pixel view
    zmin = float(np.min(Z_disp_left) - 1.0)
    zmax = float(np.max(Z_disp_left) + 1.0)

    # Flip v to match MATLAB's axis ij (origin top-left)
    v_plot = img_height_px - v_px

    # Pixel-frame points
    pts_px = np.column_stack([u_px, v_plot, Z_disp_left])
    cloud_px = pv.PolyData(pts_px)
    cloud_px["Z_mm"] = z_mm          # colour by (tilt-corrected) depths

    # ---------- Plane and bounds (at bottom of display Z-range) ----------
    plane_px = pv.Plane(
        center=(img_width_px / 2.0, img_height_px / 2.0, zmin),
        direction=(0, 0, 1),
        i_size=img_width_px,
        j_size=img_height_px,
        i_resolution=1,
        j_resolution=1,
    )

    # Invisible cube to enforce bounds
    z_center_px = 0.5 * (zmin + zmax)
    z_length_px = (zmax - zmin)
    bbox_px = pv.Cube(
        center=(img_width_px / 2.0, img_height_px / 2.0, z_center_px),
        x_length=img_width_px,
        y_length=img_height_px,
        z_length=z_length_px,
    )

    # -------- Set up single-view plotter --------
    plotter = pv.Plotter(window_size=(1920, 1080))

    plotter.add_axes(line_width=2)
    plotter.show_grid(
        xtitle="u_px",
        ytitle="v_px (image coords)",
        ztitle="Z (mm)",
    )

    plotter.add_mesh(bbox_px, opacity=0.0)  # enforce bounds

    # Image plane
    if image_path is not None:
        tex = pv.read_texture(image_path)
        tex.flip_y()
        plotter.add_mesh(
            plane_px,
            texture=tex,
            show_edges=True,
            opacity=0.9,
        )
    else:
        plotter.add_mesh(
            plane_px,
            style="wireframe",
            opacity=0.3,
            show_edges=True,
        )

    # SLI points above the image (with exaggerated Z)
    plotter.add_mesh(
        cloud_px,
        render_points_as_spheres=True,
        point_size=7,
        scalars="Z_mm",          # colour by tilt-corrected depth
        cmap="viridis",
        scalar_bar_args={"title": "Depth Z (mm)"},
    )

    # Camera above looking down
    center_px = (img_width_px / 2.0, img_height_px / 2.0, z_center_px)
    cam_height_px = zmax + (zmax - zmin)
    plotter.camera_position = [
        (center_px[0], center_px[1], cam_height_px),
        center_px,
        (0, 1, 0),
    ]

    plotter.add_title("Pixel frame: (u_px, v_px, Z_mm)", font_size=14)
    plotter.show()


if __name__ == "__main__":
    obstacle_csv = Config.SLI_CSV_GLOB_OBSTACLE
    #noObstacle_csv = Config.SLI_CSV_GLOB_NO_OBSTACLE

    img_obstacle = Config.IMG_OBSTACLE
    #img_noObstacle = Config.IMG_NO_OBSTACLE

    TILT_DEG = 5  # or 15, or whatever you decided

    # --- create calibrated CSVs in the script directory ---
    #export_calibrated_sli_csv(noObstacle_csv,output_name="calibrated_sli_noObstacle.csv",tilt_deg=TILT_DEG,calibrateTilt=True)

    #export_calibrated_sli_csv(obstacle_csv,output_name="sli_obstacle.csv",tilt_deg=TILT_DEG,calibrateTilt=True)

    # (optional) visualize using your existing function, now reading the *original*
    # while knowing the CSV versions on disk are also calibrated.
    #plot_sli_points(noObstacle_csv,image_path=img_noObstacle,tilt_deg=TILT_DEG,calibrateTilt=False)

    plot_sli_points(
        obstacle_csv,
        image_path=img_obstacle,
        tilt_deg=TILT_DEG,
        calibrateTilt=False,
    )