import argparse
from pathlib import Path
import numpy as np

from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import normalize_uint8
from .sfs_horn import (
    shape_from_shading_multilight_const_albedo,
    build_led_dirs_measured,
)
from .overlay_sli_point import load_sli_csv          # <-- add this
from .config import Config


def main(
        input_glob_or_folder: str = Config.DEFAULT_INPUT_GLOB,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR,
):
    # Ensure output directory exists
    Config.ensure_dir(output_dir)

    print(f"Input glob: {input_glob_or_folder}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------ #
    # 1. Load the 6 input images (H, W, 6)
    # ------------------------------------------------------------------ #
    I, files = load_pngs(input_glob_or_folder)
    print(f"Loaded {I.shape[-1]} images:")
    for f in files:
        print(f" - {f}")

    # Convert to float in [0, 1]
    I = I.astype(np.float32)
    if I.max() > 1.0 + 1e-3:
        I /= 255.0

    # Save composite for reference (not used directly in SFS now)
    E = I.mean(axis=-1)
    save_image(
        normalize_uint8(E),
        str(Path(output_dir) / "sfs_input_composite.png"),
    )

    # Multi-light SFS using all 6 images + measured LED directions
    L = build_led_dirs_measured()

    z, p, q = shape_from_shading_multilight_const_albedo(
        I,
        light_dirs=L,
        rho=1.0,        # fixed albedo
        n_iters=150,    # you can tweak
        step=0.01,
        lam_smooth=1.5, # tradeoff: larger = smoother
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # 4. Calibrate depth with SLI points (global scale + offset in mm)
    # ------------------------------------------------------------------ #

    # Path to your SLI CSV (same one you used before)
    #sli_csv = Path("PIXL_Images/CalData/PIXL_040mm_dist/WithObstacle/"A251110_13410908_SLI_points.csv")
    sli_csv = Path("PIXL_Images/CalData/PIXL_040mm_dist/NoObstacle/"
                   "A251110_13373123_SLI_points.csv")

    try:
        u_px, v_px, Z_sli_raw = load_sli_csv(sli_csv)
    except FileNotFoundError:
        print(f"WARNING: SLI CSV not found at {sli_csv}, skipping calibration.")
        z_cal = z.copy()
    else:
        # Convert SLI depth from µm → mm
        Z_sli_mm = Z_sli_raw.astype(np.float32) / 1000.0

        H, W = z.shape

        # MATLAB 1-based → Python 0-based indices
        u_idx = np.clip((u_px - 1).astype(int), 0, W - 1)
        v_idx = np.clip((v_px - 1).astype(int), 0, H - 1)

        # Sample SFS depth at SLI locations
        z_sfs_pts = z[v_idx, u_idx]

        valid = np.isfinite(z_sfs_pts)
        if np.sum(valid) < 2:
            print("WARNING: Not enough valid SLI ↔ SfS correspondences; "
                  "skipping calibration.")
            z_cal = z.copy()
        else:
            zv = z_sfs_pts[valid].astype(np.float32)
            Zv = Z_sli_mm[valid].astype(np.float32)

            # Use first valid point as reference
            z_ref = zv[0]
            Z_ref = Zv[0]

            dz = zv - z_ref
            dZ = Zv - Z_ref

            denom = float(np.sum(dz * dz))
            if denom < 1e-8:
                print("WARNING: SFS depths at SLI points are too similar; "
                      "using scale=1.0.")
                a = 1.0
            else:
                a = float(np.sum(dZ * dz) / denom)

            b = float(Z_ref - a * z_ref)

            print(f"SfS–SLI calibration: scale a = {a:.6f}, offset b = {b:.6f} mm")

            # Calibrated depth in mm
            z_cal = a * z + b
    # ------------------------------------------------------------------ #
    # 5. Save outputs
    # ------------------------------------------------------------------ #
    depth_base = Path(output_dir) / "sfs_depth_relative"
    save_image(
        normalize_uint8(np.nan_to_num(z, nan=0.0)),
        str(depth_base.with_suffix(".png")),
    )
    save_float_array(z, str(depth_base.with_suffix(".npy")), format="npy")
    save_float_array(z, str(depth_base.with_suffix(".pfm")), format="pfm")

    depth_cal_base = Path(output_dir) / "sfs_depth_calibrated_mm"
    save_image(
        normalize_uint8(np.nan_to_num(z_cal, nan=0.0)),
        str(depth_cal_base.with_suffix(".png")),
    )
    save_float_array(z_cal, str(depth_cal_base.with_suffix(".npy")), format="npy")
    save_float_array(z_cal, str(depth_cal_base.with_suffix(".pfm")), format="pfm")

    # Slopes
    save_float_array(p, str(Path(output_dir) / "sfs_p.npy"), format="npy")
    save_float_array(q, str(Path(output_dir) / "sfs_q.npy"), format="npy")

    print("Shape-from-shading + SLI calibration finished.")
    print(f"Wrote SFS outputs to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horn Shape-from-Shading Pipeline")
    parser.add_argument(
        "--input",
        default=Config.DEFAULT_INPUT_GLOB,
        help="Input glob or folder (expects 6 images)",
    )
    parser.add_argument(
        "--output",
        default=Config.DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    main(
        input_glob_or_folder=args.input,
        output_dir=args.output,
    )
