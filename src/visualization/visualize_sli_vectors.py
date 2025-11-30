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

def plot_sli_points_pyvista_dual(
    csv_path,
    img_width_px=752,
    img_height_px=580,
    image_path=None,
    tilt_deg=0.0,         # tilt angle IN DEGREES
    calibrateTilt=False,  # <---- NEW FLAG
):
    # -------- Load data --------
    u_px, v_px, X_mm, Y_mm, Z_mm = load_sli_points(csv_path)

    # ---- Apply tilt correction only if calibrateTilt=True ----
    if calibrateTilt and tilt_deg != 0.0:
        theta = np.deg2rad(tilt_deg)
        m = np.tan(theta)                # dZ/dY slope
        Y0 = float(np.mean(Y_mm))        # preserve mean Z level
        Z_mm = Z_mm - m * (Y_mm - Y0)    # tilt correction

    # How much to exaggerate depth in the LEFT (pixel) view
    Z_SCALE_LEFT = 5.0

    # Flip Z for visualization so closest depth is highest
    Z_disp = -Z_mm  # used for geometry; Z_mm still used for colouring

    # ---- LEFT VIEW: use scaled Z for display ----
    z_center_left = float(np.mean(Z_disp))
    Z_disp_left = (Z_disp - z_center_left) * Z_SCALE_LEFT + z_center_left

    # Z-limits left/right
    zmin_left = float(np.min(Z_disp_left) - 1.0)
    zmax_left = float(np.max(Z_disp_left) + 1.0)

    zmin_right = float(np.min(Z_disp) - 1.0)
    zmax_right = float(np.max(Z_disp) + 1.0)

    # Flip v to match MATLAB's axis ij (origin top-left)
    v_plot = img_height_px - v_px

    # Pixel-frame points (left view)
    pts_px = np.column_stack([u_px, v_plot, Z_disp_left])
    cloud_px = pv.PolyData(pts_px)
    cloud_px["Z_mm"] = Z_mm          # colour by corrected depths

    # Camera-frame points (right view)
    pts_mm = np.column_stack([X_mm, Y_mm, Z_disp])
    cloud_mm = pv.PolyData(pts_mm)
    cloud_mm["Z_mm"] = Z_mm

    # ---------- PLANES (at bottom of display Z-range) ----------

    # Left: pixel plane (uses exaggerated Z range)
    plane_px = pv.Plane(
        center=(img_width_px / 2.0, img_height_px / 2.0, zmin_left),
        direction=(0, 0, 1),
        i_size=img_width_px,
        j_size=img_height_px,
        i_resolution=1,
        j_resolution=1,
    )

    # Right: camera ground plane (uses true Z range)
    size_x = float(np.max(X_mm) - np.min(X_mm))
    size_y = float(np.max(Y_mm) - np.min(Y_mm))
    plane_mm = pv.Plane(
        center=(float(np.mean(X_mm)), float(np.mean(Y_mm)), zmin_right),
        direction=(0, 0, 1),
        i_size=size_x,
        j_size=size_y,
        i_resolution=10,
        j_resolution=10,
    )

    # Invisible cubes to enforce bounds in each subplot
    z_center_px = 0.5 * (zmin_left + zmax_left)
    z_length_px = (zmax_left - zmin_left)
    bbox_px = pv.Cube(
        center=(img_width_px / 2.0, img_height_px / 2.0, z_center_px),
        x_length=img_width_px,
        y_length=img_height_px,
        z_length=z_length_px,
    )

    z_center_mm = 0.5 * (zmin_right + zmax_right)
    z_length_mm = (zmax_right - zmin_right)
    bbox_mm = pv.Cube(
        center=(float(np.mean(X_mm)), float(np.mean(Y_mm)), z_center_mm),
        x_length=size_x,
        y_length=size_y,
        z_length=z_length_mm,
    )

    # -------- Set up plotter with 2 subplots --------
    plotter = pv.Plotter(shape=(1, 2), window_size=(1920, 1080))

    # ============================================================
    # LEFT VIEW: Pixel frame (u_px, v_px, Z_disp_left) + image
    # ============================================================
    plotter.subplot(0, 0)
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
        point_size=15,
        scalars="Z_mm",          # colour by tilt-corrected depth
        cmap="viridis",
        scalar_bar_args={"title": "Depth Z (mm)"},
    )

    # Camera above looking down
    center_px = (img_width_px / 2.0, img_height_px / 2.0, z_center_px)
    cam_height_px = zmax_left + (zmax_left - zmin_left)
    plotter.camera_position = [
        (center_px[0], center_px[1], cam_height_px),
        center_px,
        (0, 1, 0),
    ]

    plotter.add_title("Pixel frame: (u_px, v_px, Z_mm)", font_size=14)

    # ============================================================
    # RIGHT VIEW: Camera frame (X_mm, Y_mm, Z_disp) + image
    # ============================================================
    plotter.subplot(0, 1)
    plotter.add_axes(line_width=2)
    plotter.show_grid(
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
    )

    plotter.add_mesh(bbox_mm, opacity=0.0)

    if image_path is not None:
        tex_cam = pv.read_texture(image_path)
        tex_cam.flip_y()
        plotter.add_mesh(
            plane_mm,
            texture=tex_cam,
            show_edges=True,
            opacity=0.9,
        )
    else:
        plotter.add_mesh(
            plane_mm,
            style="wireframe",
            opacity=0.3,
            show_edges=True,
        )

    plotter.add_mesh(
        cloud_mm,
        render_points_as_spheres=True,
        point_size=15,
        scalars="Z_mm",
        cmap="viridis",
        scalar_bar_args={"title": "Depth Z (mm)"},
    )

    center_mm = (float(np.mean(X_mm)), float(np.mean(Y_mm)), z_center_mm)
    cam_height_mm = zmax_right + (zmax_right - zmin_right)
    plotter.camera_position = [
        (center_mm[0], center_mm[1], cam_height_mm),
        center_mm,
        (0, 1, 0),
    ]

    plotter.add_title("Camera frame: (X_mm, Y_mm, Z_mm)", font_size=14)

    # -------- Show interactive window --------
    plotter.show()


if __name__ == "__main__":
    path_Obstacle = Config.SLI_CSV_GLOB_OBSTACLE
    path_noObstacle = Config.SLI_CSV_GLOB_NO_OBSTACLE

    img_path_obstacle = Config.IMG_OBSTACLE
    img_path_noObstacle = Config.IMG_NO_OBSTACLE

    # hard-coded tilt of 10 degrees
    TILT_DEG = 10
    plot_sli_points_pyvista_dual(path_noObstacle,image_path=img_path_noObstacle,tilt_deg=TILT_DEG, calibrateTilt=False)
    plot_sli_points_pyvista_dual(path_noObstacle,image_path=img_path_noObstacle,tilt_deg=TILT_DEG, calibrateTilt=True)
    TILT_DEG = 12.5
    #plot_sli_points_pyvista_dual(path_noObstacle,image_path=img_path_noObstacle,tilt_deg=TILT_DEG, calibrateTilt=True)
    TILT_DEG = 15
    plot_sli_points_pyvista_dual(path_noObstacle, image_path=img_path_noObstacle, tilt_deg=TILT_DEG, calibrateTilt=True)
    #plot_sli_points_pyvista_dual(path_Obstacle,image_path=img_path_obstacle,tilt_deg=TILT_DEG)