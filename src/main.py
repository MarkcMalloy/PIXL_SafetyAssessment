from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .image_io import load_pngs, save_image, save_float_array
from src.pipeline.preprocessing import normalize_uint8
from src.pipeline.sfs_horn import (
    shape_from_shading_multilight_const_albedo,
    build_led_dirs_measured,
)
from src.pipeline.overlay_sli_point import load_sli_csv
from .config import Config


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
    n_iters: int = 150,
    step: float = 0.01,
    lam_smooth: float = 2.0,
):
    """
    Run the full SfS + SLI calibration pipeline for one dataset.

    Parameters
    ----------
    name : str
        Label printed in the logs (e.g. "NO_OBSTACLE", "OBSTACLE").
    input_glob : str
        Glob for the 6 PNG images.
    sli_csv_glob : str
        Glob for the SLI CSV file(s); first match is used.
    output_dir : str
        Output directory for this dataset.
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

    # Save composite for reference (not used directly in SFS now)
    E = I.mean(axis=-1)
    save_image(
        normalize_uint8(E),
        str(out_dir / "sfs_input_composite.png"),
    )

    # ------------------------------------------------------------------ #
    # 2. Multi-light SFS using all 6 images + measured LED directions
    # ------------------------------------------------------------------ #
    L = build_led_dirs_measured()

    z, p, q = shape_from_shading_multilight_const_albedo(
        I,
        light_dirs=L,
        #rho=0.55,        # fixed albedo
        rho=1,        # fixed albedo
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # 3. Calibrate depth with SLI points (global scale + offset in mm)
    # ------------------------------------------------------------------ #
    sli_csv = _pick_first_csv_from_glob(sli_csv_glob)
    if sli_csv is None:
        print(f"WARNING: No SLI CSV found for glob {sli_csv_glob}, "
              f"skipping calibration for dataset {name}.")
        z_cal = z.copy()
    else:
        print(f"Using SLI CSV: {sli_csv}")
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

                print(f"[{name}] SfS–SLI calibration: "
                      f"scale a = {a:.6f}, offset b = {b:.6f} mm")

                # Calibrated depth in mm
                z_cal = a * z + b

    # ------------------------------------------------------------------ #
    # 4. Save outputs
    # ------------------------------------------------------------------ #
    depth_base = out_dir / "sfs_depth_relative"
    save_image(
        normalize_uint8(np.nan_to_num(z, nan=0.0)),
        str(depth_base.with_suffix(".png")),
    )
    save_float_array(z, str(depth_base.with_suffix(".npy")), format="npy")
    save_float_array(z, str(depth_base.with_suffix(".pfm")), format="pfm")

    depth_cal_base = out_dir / "sfs_depth_calibrated_mm"
    save_image(
        normalize_uint8(np.nan_to_num(z_cal, nan=0.0)),
        str(depth_cal_base.with_suffix(".png")),
    )
    save_float_array(z_cal, str(depth_cal_base.with_suffix(".npy")), format="npy")
    save_float_array(z_cal, str(depth_cal_base.with_suffix(".pfm")), format="pfm")

    # Slopes
    save_float_array(p, str(out_dir / "sfs_p.npy"), format="npy")
    save_float_array(q, str(out_dir / "sfs_q.npy"), format="npy")

    print(f"[{name}] Shape-from-shading + SLI calibration finished.")
    print(f"[{name}] Wrote SFS outputs to: {out_dir.resolve()}")


def main():
    """
    Run the pipeline for BOTH datasets:
      - NO_OBSTACLE
      - OBSTACLE
    using paths from Config.
    """
    run_dataset(
        name="NO_OBSTACLE",
        input_glob=Config.DEFAULT_INPUT_GLOB_NO_OBSTACLE,
        sli_csv_glob=Config.DEFAULT_SLI_CSV_GLOB_NO_OBSTACLE,
        output_dir=Config.OUTPUT_DIR_DEPTH_NO_OBSTACLE,
    )

    run_dataset(
        name="OBSTACLE",
        input_glob=Config.DEFAULT_INPUT_GLOB_OBSTACLE,
        sli_csv_glob=Config.DEFAULT_SLI_CSV_GLOB_OBSTACLE,
        output_dir=Config.OUTPUT_DIR_DEPTH_OBSTACLE,
    )


if __name__ == "__main__":
    # Keep a minimal CLI hook, but ignore arguments for now and just run both.
    parser = argparse.ArgumentParser(
        description="Horn Shape-from-Shading Pipeline (both datasets)")
    _ = parser.parse_args()
    main()
