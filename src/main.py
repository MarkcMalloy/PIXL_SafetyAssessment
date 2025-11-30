from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.image_io import load_pngs, save_image, save_float_array
from src.pipeline.preprocessing import normalize_uint8, otsu_on_max
from src.pipeline.sfs_horn import (
    shape_from_shading_multilight_const_albedo,
    build_led_dirs_measured,
    build_light_dirs_tilted,
    shape_from_shading_multilight,
)
from src.pipeline.visualization import save_depth_plot
from src.config import Config

def _fit_scale_offset(
    z: np.ndarray,
    Z_sli_mm: np.ndarray,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    name: str,
):
    """
    Given a relative SfS height map z and SLI depths Z_sli_mm at pixel indices
    (u_idx, v_idx), fit a linear mapping:

        Z_mm ≈ a * z + b

    Returns (a, b). Does NOT modify z.
    """
    H, W = z.shape
    z_sfs_pts = z[v_idx, u_idx]

    valid = np.isfinite(z_sfs_pts)
    if np.sum(valid) < 2:
        print("WARNING: Not enough valid SLI ↔ SfS correspondences; "
              "using a=1, b=0.")
        return 1.0, 0.0

    zv = z_sfs_pts[valid].astype(np.float32)
    Zv = Z_sli_mm[valid].astype(np.float32)

    # Use first valid point as reference to reduce numerical issues
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

    print(f"[{name}] SfS–SLI calibration: scale a = {a:.6f}, offset b = {b:.6f} mm")
    return a, b


def _load_sli_points_for_calibration(csv_path: Path):
    """
    Read the calibrated SLI CSV in the same way as visualize_sli_vectors.py:

      - u_px, v_px : pixel coordinates in the image (0..W-1, 0..H-1)
      - Z_mm       : depth in millimetres

    Only the Z column is needed for calibration, X/Y camera coords are ignored.
    """
    df = pd.read_csv(csv_path)

    u_px = df["u_px"].to_numpy(dtype=float)
    v_px = df["v_px"].to_numpy(dtype=float)

    Z_um = df["Z_um"].to_numpy(dtype=float)
    Z_mm = Z_um / 1000.0  # µm → mm

    return u_px, v_px, Z_mm

def _pick_first_csv_from_glob(csv_glob: str) -> Path | None:
    """
    Given a glob like ".../WithObstacle/*.csv", return the first matching file
    (sorted). Returns None if there is no match.
    """
    pattern = Path(csv_glob)
    csv_dir = pattern.parent
    csv_pat = pattern.name
    matches = sorted(csv_dir.glob(csv_pat))
    if not matches:
        return None
    return matches[0]

def run_dataset(
    name: str,
    input_glob: str,
    sli_csv_glob: str,
    output_dir: str,
    n_iters: int = 200,
    step: float = 0.04,
    lam_smooth: float = 0.01,
):
    """
    Run the full SfS pipeline for one dataset, with optional *global*
    SLI-based scale calibration. The SLI points DO NOT warp the shape;
    they only set a and b in

        Z_mm ≈ a * z + b

    so that depth is in millimetres.
    """

    print("=" * 80)
    print(f"Running dataset: {name}")
    print(f"  Images glob : {input_glob}")
    print(f"  SLI CSV glob: {sli_csv_glob}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 80)

    # Ensure output directory exists
    Config.ensure_dir(output_dir)
    out_dir = Path(output_dir)

    # ------------------------------------------------------------------ #
    # 1. Load the 6 input images (H, W, 6)
    # ------------------------------------------------------------------ #
    I, files = load_pngs(input_glob)
    print(f"Loaded {I.shape[-1]} images:")
    for f in files:
        print(f" - {f}")

    # Convert to float in [0, 1]
    I = I.astype(np.float32)
    if I.max() > 1.0 + 1e-3:
        I /= 255.0

    # Save composite for reference
    E = I.mean(axis=-1)
    save_image(
        normalize_uint8(E),
        str(out_dir / "sfs_input_composite.png"),
    )

    # ------------------------------------------------------------------ #
    # 2. Multi-light SfS using all 6 images + tilted LED directions
    # ------------------------------------------------------------------ #
    L = build_light_dirs_tilted()
    z, p, q, rho = shape_from_shading_multilight(
        I,
        light_dirs=L,
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        verbose=True,
    )
    print("Rho is:", rho)

    # This is the *pure* SfS shape (arbitrary scale & offset)
    z_rel = z.copy()

    # ------------------------------------------------------------------ #
    # 3. Optional: global SfS → SLI scale (no warping)
    # ------------------------------------------------------------------ #
    sli_csv = _pick_first_csv_from_glob(sli_csv_glob)
    if sli_csv is None:
        print(f"WARNING: No SLI CSV found for glob {sli_csv_glob}, "
              f"skipping calibration for dataset {name}.")
        a, b = 1.0, 0.0
        z_cal = z_rel.copy()
    else:
        print(f"Using SLI CSV: {sli_csv}")
        try:
            # Same interpretation as visualize_sli_vectors.py
            u_px, v_px, Z_sli_mm = _load_sli_points_for_calibration(sli_csv)
        except FileNotFoundError:
            print(f"WARNING: SLI CSV not found at {sli_csv}, skipping calibration.")
            a, b = 1.0, 0.0
            z_cal = z_rel.copy()
        else:
            H, W = z_rel.shape

            # Pixel indices: u_px, v_px already in [0,W-1] / [0,H-1]
            # Flip v to match the SfS image frame (same convention as
            # view_depth_calibrated and visualize_sli_vectors).
            u_sfs = u_px
            v_sfs = (H - 1) - v_px

            u_idx = np.clip(np.rint(u_sfs).astype(int), 0, W - 1)
            v_idx = np.clip(np.rint(v_sfs).astype(int), 0, H - 1)

            # Fit *global* scale and offset, do not modify z_rel
            a, b = _fit_scale_offset(z_rel, Z_sli_mm, u_idx, v_idx, name)

            # Calibrated depth in mm (same shape, just re-scaled)
            z_cal = a * z_rel + b

    # ------------------------------------------------------------------ #
    # 4. Save outputs
    # ------------------------------------------------------------------ #
    # Relative SfS depth (no SLI info)
    depth_base = out_dir / "sfs_depth_relative"
    z_rel_safe = np.nan_to_num(z_rel, nan=0.0)
    save_image(
        normalize_uint8(z_rel_safe),
        str(depth_base.with_suffix(".png")),
    )
    save_float_array(z_rel_safe, str(depth_base.with_suffix(".npy")))

    # Calibrated depth in mm (only scaled/offset)
    depth_cal_base = out_dir / "sfs_depth_calibrated_mm"
    z_cal_safe = np.nan_to_num(z_cal, nan=0.0)
    save_image(
        normalize_uint8(z_cal_safe),
        str(depth_cal_base.with_suffix(".png")),
    )
    save_float_array(z_cal_safe, str(depth_cal_base.with_suffix(".npy")))

    # Slopes and albedo for inspection / future use
    save_float_array(p, str(out_dir / "sfs_p.npy"))
    save_float_array(q, str(out_dir / "sfs_q.npy"))
    save_float_array(rho, str(out_dir / "sfs_rho.npy"))

    print(f"[{name}] Shape-from-shading finished. "
          f"a = {a:.6f}, b = {b:.6f} mm")
    print(f"[{name}] Wrote SFS outputs to: {out_dir}")



def main():
    """
    Run the pipeline for BOTH datasets:
      - NO_OBSTACLE
      - OBSTACLE
    using paths from Config.
    """

    # Run these if you are just using the SLI csv from the testData
    #run_dataset(name="NO_OBSTACLE",input_glob=Config.DEFAULT_INPUT_GLOB_NO_OBSTACLE,sli_csv_glob=Config.DEFAULT_SLI_CSV_GLOB_NO_OBSTACLE,output_dir=Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE,)
    #run_dataset(name="OBSTACLE",input_glob=Config.DEFAULT_INPUT_GLOB_OBSTACLE,sli_csv_glob=Config.DEFAULT_SLI_CSV_GLOB_OBSTACLE,output_dir=Config.OUTPUT_DIR_DEPTH_OBSTACLE,)

    # Run these if you are just using the calibrated SLI csv after having run visualize_sli_vectors

    #run_dataset(name="NO_OBSTACLE_SLICal", input_glob=Config.DEFAULT_INPUT_GLOB_NO_OBSTACLE,sli_csv_glob=Config.SLI_NO_OBSTACLE_CSV_CALIBRATED, output_dir=Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE, )
    run_dataset(name="OBSTACLE_SLICal", input_glob=Config.DEFAULT_INPUT_GLOB_OBSTACLE,sli_csv_glob=Config.SLI_OBSTACLE_CSV_CALIBRATED, output_dir=Config.OUTPUT_DIR_DEPTH_OBSTACLE, )
    #run_dataset(name="OBSTACLE_SLI", input_glob=Config.DEFAULT_INPUT_GLOB_OBSTACLE,sli_csv_glob=Config.SLI_CSV_GLOB_OBSTACLE, output_dir=Config.OUTPUT_DIR_DEPTH_OBSTACLE, )



if __name__ == "__main__":
    # Keep a minimal CLI hook, but ignore arguments for now and just run both.
    parser = argparse.ArgumentParser(
        description="Horn Shape-from-Shading Pipeline (both datasets)")
    _ = parser.parse_args()
    main()
