import argparse
from pathlib import Path
import numpy as np
import cv2  # <- needed for bilateralFilter
from scipy.interpolate import Rbf

from .config import Config
from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import otsu_on_max, quantile_mask, normalize_uint8
from src.photometric_stereo import (
    # basic ring model
    build_light_dirs_tilted,     # ring + camera tilt
    # point-light (tilt + optional offset)
    build_light_dirs_point_measured,  # point-light from measured LED positions
    solve_photometric_stereo,
    solve_photometric_stereo_uniform_albedo,
)
from .depth_estimation import normals_to_depth
from .visualization import save_normals_rgb, save_shadow_maps, save_depth_plot
from .overlay_sli_point import load_sli_csv, overlay_sli_points


def main(
        input_glob_or_folder: str = Config.DEFAULT_INPUT_GLOB,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR,
        albedo_dir: str = Config.OUTPUT_DIR_ALBEDO,
        composite_dir: str = Config.OUTPUT_DIR_COMPOSITES,
        depth_dir: str = Config.OUTPUT_DIR_DEPTH,
        mask_dir: str = Config.OUTPUT_DIR_MASKS,
        norm_dir: str = Config.OUTPUT_DIR_NORMALIZATION,
        shadow_dir: str = Config.OUTPUT_DIR_SHADOWS,
        use_otsu: bool = True,
        mask_quantile: float = Config.DEFAULT_MASK_QUANTILE
):
    # Ensure output directories exist
    for d in [output_dir, albedo_dir, composite_dir, depth_dir, mask_dir, norm_dir, shadow_dir]:
        Config.ensure_dir(d)

    print(f"Input glob: {input_glob_or_folder}")
    print(f"Output directory: {output_dir}")
    overlay_sli_points()
    # Load images
    I, files = load_pngs(input_glob_or_folder)

    # Mask
    if use_otsu:
        mask, Imax = otsu_on_max(I)
    else:
        Imax = I.max(axis=-1)
        mask = quantile_mask(I, mask_quantile)

    # Per-light gain calibration (same gains used for all L variants)
    mean_intensity = [I[..., i].mean() for i in range(I.shape[-1])]
    s = np.array(mean_intensity) / np.mean(mean_intensity)
    I_cal = I / s[None, None, :]

    # --- Compute normals for each light model and save for comparison ---
    # --- Compute normals for each light model and save for comparison ---
    variants = {
        "tilted":   build_light_dirs_tilted,
        "measured": build_light_dirs_point_measured,
    }

    normals_by_variant = {}
    albedo_by_variant = {}

    for name, builder in variants.items():
        L = builder()
        albedo_v, n_v = solve_photometric_stereo(I_cal, L, mask)
        n_uniform_albedo_v = solve_photometric_stereo_uniform_albedo(I_cal, L, mask)
        save_normals_rgb(n_v, str(Path(norm_dir) / f"normals_{name}.png"))
        save_normals_rgb(
            n_uniform_albedo_v,
            str(Path(norm_dir) / f"normals_uniform_albedo_{name}.png"),
        )
        normals_by_variant[name] = n_v
        albedo_by_variant[name] = albedo_v

    # --- Choose one variant (TILTED) to produce the rest of the outputs ---
    n = normals_by_variant["tilted"]
    albedo = albedo_by_variant["tilted"]

    # --- Depth from normals (relative units) ---
    n_smooth = cv2.bilateralFilter(n.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3)
    z = normals_to_depth(n_smooth, mask)   # relative depth (arbitrary units)

    # --- Calibrate depth with SLI points (linear fit, SLI in mm) ---
    # Load SLI CSV using our helper: returns u_px, v_px, Z_sli_raw
    # Load SLI CSV
    sli_csv = Path("PIXL_Images/CalData/PIXL_040mm_dist/WithObstacle/A251110_13410908_SLI_points.csv")
    u_px, v_px, Z_sli_raw = load_sli_csv(sli_csv)

    # Convert SLI depth from micrometers → millimeters
    Z_sli_mm = Z_sli_raw.astype(np.float32) / 1000.0

    H, W = z.shape

    # Convert MATLAB 1-based → Python 0-based indexing
    u_idx = np.clip((u_px - 1).astype(int), 0, W - 1)
    v_idx = np.clip((v_px - 1).astype(int), 0, H - 1)

    # Sample SfS depth at the SLI pixel locations
    z_sfs_pts = z[v_idx, u_idx]

    # Keep only valid SLI samples
    valid_pts = np.isfinite(z_sfs_pts) & (mask[v_idx, u_idx] > 0)
    if np.sum(valid_pts) < 2:
        print("Warning: Not enough valid SLI ↔ SfS correspondences.")
        z_cal = z.copy()
    else:
        u_valid = u_idx[valid_pts].astype(float)
        v_valid = v_idx[valid_pts].astype(float)
        Z_sli_valid = Z_sli_mm[valid_pts]
        z_sfs_valid = z_sfs_pts[valid_pts]

        # Compute offset at each SLI point: Δ_i = Z_sli_mm_i − z_sfs_i
        offsets = Z_sli_valid - z_sfs_valid

        # Build smooth 2-D offset field Δ(x, y) using radial basis interpolation
        print(f"Interpolating dense offset field using {len(offsets)} SLI points…")
        rbf = Rbf(u_valid, v_valid, offsets, function='multiquadric', smooth=2.0)

        yy, xx = np.mgrid[0:H, 0:W]
        offset_field = rbf(xx, yy)   # same shape as z

        # Apply dense correction field
        z_cal = z + offset_field

    # ---------------- Save calibrated depth in mm ----------------
    save_image(
        normalize_uint8(np.nan_to_num(z_cal, nan=0.0)),
        str(Path(depth_dir) / "depth_calibrated.png"),
    )
    save_float_array(z_cal, str(Path(depth_dir) / "depth_calibrated.npy"), format="npy")
    save_float_array(z_cal, str(Path(depth_dir) / "depth_calibrated.pfm"), format="pfm")

    # --- Visualization-only: subtract mean to reveal true 3-D shape ---
    valid = np.isfinite(z_cal) & (mask > 0)
    z_cal_vis = z_cal.copy()
    z_cal_vis[valid] -= np.nanmean(z_cal_vis[valid])

    save_depth_plot(z_cal_vis, mask, str(Path(depth_dir) / "depth_calibrated_3d.png"))

    ###


    # Save other outputs
    save_image(normalize_uint8(albedo), str(Path(albedo_dir) / "albedo.png"))
    save_image(normalize_uint8(np.nan_to_num(z, nan=0.0)), str(Path(depth_dir) / "depth.png"))
    save_float_array(z, str(Path(depth_dir) / "depth.npy"), format="npy")
    save_float_array(z, str(Path(depth_dir) / "depth.pfm"), format="pfm")
    save_depth_plot(z, mask, str(Path(depth_dir) / "depth_3d.png"))

    # Shadows from the chosen variant
    L_measured = build_light_dirs_point_measured()
    save_shadow_maps(n, L_measured, mask, shadow_dir)

    # Mask & composite
    save_image(mask * 255, str(Path(mask_dir) / "mask.png"))
    save_image(normalize_uint8(Imax), str(Path(composite_dir) / "composite_max.png"))

    # Summary
    print("Processed files (first 6):")
    for f in files[:6]:
        print(f" - {f}")
    print(f"Wrote outputs to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photometric Stereo Pipeline")
    parser.add_argument("--input", default=Config.DEFAULT_INPUT_GLOB, help="Input glob or folder")
    parser.add_argument("--output", default=Config.DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--albedo-dir", default=Config.OUTPUT_DIR_ALBEDO, help="Albedo output directory")
    parser.add_argument("--composite-dir", default=Config.OUTPUT_DIR_COMPOSITES, help="Composites output directory")
    parser.add_argument("--depth-dir", default=Config.OUTPUT_DIR_DEPTH, help="Depth output directory")
    parser.add_argument("--mask-dir", default=Config.OUTPUT_DIR_MASKS, help="Masks output directory")
    parser.add_argument("--norm-dir", default=Config.OUTPUT_DIR_NORMALIZATION, help="Normalizations output directory")
    parser.add_argument("--shadow-dir", default=Config.OUTPUT_DIR_SHADOWS, help="Shadows output directory")
    parser.add_argument("--no-otsu", action="store_false", dest="use_otsu", help="Use quantile mask instead of Otsu")
    parser.add_argument("--mask-quantile", type=float, default=Config.DEFAULT_MASK_QUANTILE, help="Quantile for mask")
    args = parser.parse_args()

    main(
        input_glob_or_folder=args.input,
        output_dir=args.output,
        albedo_dir=args.albedo_dir,
        composite_dir=args.composite_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        norm_dir=args.norm_dir,
        shadow_dir=args.shadow_dir,
        use_otsu=args.use_otsu,
        mask_quantile=args.mask_quantile
    )