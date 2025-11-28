import numpy as np
import pandas as pd
import pyvista as pv
from src.config import Config  # keep as you had it


def load_sli_points(csv_path):
    """
    Load SLI CSV and return:
      - u_px, v_px (pixel coords)
      - X_mm, Y_mm, Z_mm (camera frame in mm)
    Adjust column names here if needed.
    """
    df = pd.read_csv(csv_path)

    # ---- Adjust these names if your CSV uses different ones ----
    u_px = df["u_px"].to_numpy(dtype=float)
    v_px = df["v_px"].to_numpy(dtype=float)

    # Spatial coordinates in micrometers → mm
    X_um = df["X_um"].to_numpy(dtype=float)
    Y_um = df["Y_um"].to_numpy(dtype=float)
    Z_um = df["Z_um"].to_numpy(dtype=float)

    X_mm = X_um / 1000.0
    Y_mm = Y_um / 1000.0
    Z_mm = Z_um / 1000.0

    return u_px, v_px, X_mm, Y_mm, Z_mm


def plot_sli_points_pyvista_dual(
    csv_path,
    img_width_px=752,
    img_height_px=580,
    image_path=None,         # <- path to 752x580 image (optional)
):
    # -------- Load data --------
    u_px, v_px, X_mm, Y_mm, Z_mm = load_sli_points(csv_path)

    # Z-limits: 1 mm below min, a little above max
    zmin = float(np.min(Z_mm) - 1.0)
    zmax = float(np.max(Z_mm) + 1.0)

    # Flip v to match MATLAB's axis ij (origin top-left)
    v_plot = img_height_px - v_px

    # Pixel-frame points (left view)
    pts_px = np.column_stack([u_px, v_plot, Z_mm])
    cloud_px = pv.PolyData(pts_px)
    cloud_px["Z_mm"] = Z_mm

    # Camera-frame points (right view)
    pts_mm = np.column_stack([X_mm, Y_mm, Z_mm])
    cloud_mm = pv.PolyData(pts_mm)
    cloud_mm["Z_mm"] = Z_mm

    # ---------- PLANES ----------

    # Pixel plane (will optionally carry a texture)
    plane_px = pv.Plane(
        center=(img_width_px / 2.0, img_height_px / 2.0, zmin),
        direction=(0, 0, 1),
        i_size=img_width_px,
        j_size=img_height_px,
        i_resolution=1,
        j_resolution=1,
    )

    # Ground plane in camera frame
    size_x = float(np.max(X_mm) - np.min(X_mm))
    size_y = float(np.max(Y_mm) - np.min(Y_mm))
    plane_mm = pv.Plane(
        center=(float(np.mean(X_mm)), float(np.mean(Y_mm)), zmin),
        direction=(0, 0, 1),
        i_size=size_x,
        j_size=size_y,
        i_resolution=10,
        j_resolution=10,
    )

    # Invisible bounding boxes to enforce Z bounds in each subplot
    bbox_px = pv.Cube(
        center=(img_width_px / 2.0, img_height_px / 2.0, (zmin + zmax) / 2.0),
        x_length=img_width_px,
        y_length=img_height_px,
        z_length=(zmax - zmin),
    )

    bbox_mm = pv.Cube(
        center=(float(np.mean(X_mm)), float(np.mean(Y_mm)), (zmin + zmax) / 2.0),
        x_length=size_x,
        y_length=size_y,
        z_length=(zmax - zmin),
    )

    # -------- Set up plotter with 2 subplots --------
    plotter = pv.Plotter(shape=(1, 2), window_size=(1920, 1080))

    # ============================================================
    # LEFT VIEW: Pixel frame (u_px, v_px, Z_mm) + optional image
    # ============================================================
    plotter.subplot(0, 0)
    plotter.add_axes(line_width=2)
    plotter.show_grid(
        xtitle="u_px",
        ytitle="v_px (image coords)",
        ztitle="Z (mm)",
    )

    # Enforce bounds
    plotter.add_mesh(bbox_px, opacity=0.0)

    # Image plane below the points
    if image_path is not None:
        tex = pv.read_texture(image_path)
        # Flip Y so that v increasing downward matches the image orientation
        tex.flip_y()
        plotter.add_mesh(
            plane_px,
            texture=tex,
            show_edges=True,
            opacity=0.9,
        )
    else:
        # Fallback: just wireframe plane
        plotter.add_mesh(
            plane_px,
            style="wireframe",
            opacity=0.3,
            show_edges=True,
        )

    # SLI points above the image plane
    plotter.add_mesh(
        cloud_px,
        render_points_as_spheres=True,
        point_size=6,
        scalars="Z_mm",
        cmap="viridis",
        scalar_bar_args={"title": "Depth Z (mm)"},
    )

    # Camera above looking down
    center_px = (img_width_px / 2.0, img_height_px / 2.0, (zmin + zmax) / 2.0)
    cam_height_px = zmax + (zmax - zmin)  # a bit above the highest point
    plotter.camera_position = [
        (center_px[0], center_px[1], cam_height_px),  # camera position
        center_px,                                    # focal point
        (0, 1, 0),                                    # up-vector
    ]

    plotter.add_title("Pixel frame: (u_px, v_px, Z_mm)", font_size=14)

    # ============================================================
    # RIGHT VIEW: Camera frame (X_mm, Y_mm, Z_mm)
    # ============================================================
    plotter.subplot(0, 1)
    plotter.add_axes(line_width=2)
    plotter.show_grid(
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
    )

    # Enforce bounds
    plotter.add_mesh(bbox_mm, opacity=0.0)

    # --- Image / ground plane in camera frame ---
    if image_path is not None:
        tex_cam = pv.read_texture(image_path)
        # Same orientation trick as in pixel frame: flip Y
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

    # SLI points in camera coordinates
    plotter.add_mesh(
        cloud_mm,
        render_points_as_spheres=True,
        point_size=6,
        scalars="Z_mm",
        cmap="viridis",
        scalar_bar_args={"title": "Depth Z (mm)"},
    )

    # Camera above looking down in camera frame too
    center_mm = (float(np.mean(X_mm)), float(np.mean(Y_mm)), (zmin + zmax) / 2.0)
    cam_height_mm = zmax + (zmax - zmin)
    plotter.camera_position = [
        (center_mm[0], center_mm[1], cam_height_mm),  # camera position
        center_mm,                                    # focal point
        (0, 1, 0),                                    # up-vector
    ]

    plotter.add_title("Camera frame: (X_mm, Y_mm, Z_mm)", font_size=14)


    # -------- Show interactive window --------
    plotter.show()


if __name__ == "__main__":
    path_Obstacle = Config.SLI_CSV_GLOB_OBSTACLE
    path_noObstacle = Config.SLI_CSV_GLOB_NO_OBSTACLE

    # TODO: replace with the actual image you want to use
    img_path_obstacle = Config.IMG_OBSTACLE
    img_path_noObstacle = Config.IMG_NO_OBSTACLE

    plot_sli_points_pyvista_dual(path_Obstacle, image_path=img_path_obstacle)
    plot_sli_points_pyvista_dual(path_noObstacle, image_path=img_path_noObstacle)
