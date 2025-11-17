from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


def load_sli_csv(csv_path: str | Path):
    """
    Load SLI CSV with columns: u,v,X,Y,Z

    Returns:
        u_raw, v_raw, X, Y, Z  as 1D numpy arrays.
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    u = data[:, 0]
    v = data[:, 1]
    X = data[:, 2]
    Y = data[:, 3]
    Z = data[:, 4]
    return u, v, X, Y, Z


def overlay_all_points(
    img_path: str | Path,
    csv_path: str | Path,
    out_path: str | Path,
    cross_half_size: int = 3,
    center_points: bool = True,
):
    """
    Overlay ALL SLI points from the CSV onto the image.

    Assumes:
        - CSV columns: u,v,X,Y,Z
        - u,v are fixed-point (Q12): pixel = raw / 4096.

    If center_points is True, the centroid of all SLI points is shifted to
    the center of the image (like the MATLAB centering behaviour).
    """
    img_path = Path(img_path)
    out_path = Path(out_path)

    # Load image
    img = Image.open(img_path)
    img_np = np.array(img)

    # Make sure we have 3-channel RGB
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    H, W, _ = img_np.shape

    # Load SLI raw data
    u_raw, v_raw, X_all, Y_all, Z_all = load_sli_csv(csv_path)
    N = u_raw.shape[0]

    print(f"Loaded {N} SLI points from {csv_path}")

    # Convert all points from fixed-point to pixel coordinates
    u_pix = u_raw / 4096.0
    v_pix = v_raw / 4096.0

    # ---- Centering step -------------------------------------------------
    if center_points and N > 0:
        # Image center in pixel coordinates
        cx_img = (W - 1) / 2.0
        cy_img = (H - 1) / 2.0

        # Centroid of the SLI pattern
        cx_pts = np.mean(u_pix)
        cy_pts = np.mean(v_pix)

        dx = cx_img - cx_pts
        dy = cy_img - cy_pts

        u_pix = u_pix + dx
        v_pix = v_pix + dy

        print(f"Centering SLI points: shift dx={dx:.2f}, dy={dy:.2f}")

    # Draw each point as a small red cross
    for i in range(N):
        col_f = u_pix[i]
        row_f = v_pix[i]

        # skip obviously bogus values
        if not np.isfinite(col_f) or not np.isfinite(row_f):
            continue

        col = int(round(col_f))
        row = int(round(row_f))

        if row < 0 or row >= H or col < 0 or col >= W:
            # outside image, skip
            continue

        # Cross centered at (row, col)
        for dr in range(-cross_half_size, cross_half_size + 1):
            r = row + dr
            if 0 <= r < H:
                img_np[r, col] = [0, 255, 0]

        for dc in range(-cross_half_size, cross_half_size + 1):
            c = col + dc
            if 0 <= c < W:
                img_np[row, c] = [0, 255, 0]

    # Save overlay
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(out_path)
    print(f"Saved overlay with all SLI points to {out_path}")


if __name__ == "__main__":
    # 👉 EDIT THESE PATHS TO YOUR FILES

    # Example: SLI CSV produced from the .unc/.sli file
    csv_path = r"PIXL_Images/CalData/PIXL_040mm_dist/WithObstacle/A251110_13410908_SLI_points.csv"

    # Example: PNG produced from the corresponding camera .unc file
    img_path = r"Output/Normalizations/normals_unifrom_albedo_basic.png"

    out_path = r"PIXL_Images\CalData\Output\40mm_NoObstacle_full\40mm_1_overlay_point1.png"

    overlay_all_points(img_path, csv_path, out_path)