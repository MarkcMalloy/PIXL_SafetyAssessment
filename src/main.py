from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
import re

from src.config import Config
from src.image_io import load_pngs, save_image, save_float_array
from src.preprocessing import otsu_on_max,otsu_max_mask2, quantile_mask, normalize_uint8
from src.photometric_stereo import (
    build_light_dirs,            # basic ring model
    build_light_dirs_tilted,     # ring + camera tilt
    build_light_dirs_point,      # point-light (tilt + optional offset)
    build_light_dirs_point_measured,  # point-light from measured LED positions
    solve_photometric_stereo,
    solve_photometric_stereo_uniform_albedo,
)
from src.depth_estimation import normals_to_depth, calibrate_depth_with_sli
from src.visualization import save_normals_rgb, save_shadow_maps, save_depth_plot
from src.illumination_calibration import apply_illumination_calibration


from src.overlay_sli_point import load_sli_csv, overlay_sli_points


D_RE = re.compile(r"_D(\d+)(?:\.png)$")

def extract_single_d_from_filename(path: str) -> int | None:
    m = D_RE.search(path.replace("\\", "/"))
    if not m:
        return None
    return int(m.group(1))

def filter_and_order_by_d(I: np.ndarray, files: list[str]):
    keep_idx = []
    d_list = []
    for i, f in enumerate(files):
        d = extract_single_d_from_filename(f)
        if d is None:
            continue
        keep_idx.append(i)
        d_list.append(d)
    I2 = I[..., keep_idx]
    files2 = [files[i] for i in keep_idx]
    return I2, files2, d_list


def main(
        input_glob_or_folder: str = Config.DEFAULT_INPUT_GLOB_40mm,
        sli_csv_glob: str = Config.DEFAULT_SLI_CSV_GLOB_40mm,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR,
        albedo_dir: str = Config.OUTPUT_DIR_ALBEDO,
        composite_dir: str = Config.OUTPUT_DIR_COMPOSITES,
        depth_dir: str = Config.OUTPUT_DIR_DEPTH,
        mask_dir: str = Config.OUTPUT_DIR_MASKS,
        norm_dir: str = Config.OUTPUT_DIR_NORMALIZATION,
        shadow_dir: str = Config.OUTPUT_DIR_SHADOWS,
        use_otsu: bool = True,
        mask_quantile: float = Config.DEFAULT_MASK_QUANTILE,
        #calibration_file: str = Config.DEFAULT_LIGHT_CALIBRATION_40mm,
        calibration_file: str = None,
        working_height: float = 40.0, #mm
        use_illumination_correction: bool = True,
        **kwargs
):
    # Ensure output directories exist
    for d in [output_dir, albedo_dir, composite_dir, depth_dir, mask_dir, norm_dir, shadow_dir]:
        Config.ensure_dir(d)

    print(f"Input glob: {input_glob_or_folder}")
    print(f"Output directory: {output_dir}")
    #overlay_sli_points()
    # Load images
    I, files = load_pngs(input_glob_or_folder)

    I, files, d_list = filter_and_order_by_d(I, files)
    print("Using files:")
    for f, d in zip(files, d_list):
        print(f" - D{d}: {f}")

    # === NEW: Apply illumination correction ===
    if use_illumination_correction and calibration_file:
        print(f"Loading calibration from {calibration_file}...")
        calibration_data = np.load(calibration_file, allow_pickle=True).item()
        
        print(f"Applying illumination correction for {working_height}mm working distance...")
        I = apply_illumination_calibration(
            I, 
            calibration_data,
            working_height=working_height,
            pixel_size=Config.DEFAULT_PIXEL_SIZE * 1000  # Convert m to mm
        )

    # Mask
    if use_otsu:
        print(f"Applying otsu threshold for {working_height}mm working distance...")

        mask, Imax = otsu_max_mask2(
            I,
            core_scale=0.5,
            morph_open_ksize=1,  # gentler opening (or 0 to disable)
            edge_dilate_ksize=7  # bigger halo for grazing edges
        )
        #mask, Imax = otsu_on_max(I)
    else:
        Imax = I.max(axis=-1)
        mask = quantile_mask(I, mask_quantile)

    # Per-light gain calibration (same gains used for all L variants)
    mean_intensity = [I[..., i].mean() for i in range(I.shape[-1])]
    s = np.array(mean_intensity) / np.mean(mean_intensity)
    I_cal = I / s[None, None, :]

    # --- Compute normals for each light model and save for comparison ---
    # --- Build measured light directions subset from PNG suffixes (D#) ---
    # d_list must already be computed from filenames in the same order as I_cal's channels.
    # -------------------------------------------------------------------------
    # Build light directions using measured LED geometry
    # -------------------------------------------------------------------------

    K = I_cal.shape[-1]
    if len(d_list) != K:
        raise ValueError(f"d_list length ({len(d_list)}) must match #images K ({K}).")

    # D1..D6 → G1..G6 (0-based index = D-1)
    D_TO_G_INDEX = {d: d - 1 for d in range(1, 7)}

    # Measured LED azimuths (MUST match led_data order in photometric_stereo.py)
    G_ANGLES_DEG = [320, 10, 60, 110, 160, 205]

    # --- Full measured point-light model (6 LEDs) ---
    L6_measured = build_light_dirs_point_measured()  # (6,3)

    # Subset measured lights in image order
    try:
        g_idx_list = [D_TO_G_INDEX[d] for d in d_list]
    except KeyError as e:
        raise ValueError(f"Unknown D index in filenames: {e}")

    L_measured_subset = np.stack([L6_measured[g] for g in g_idx_list], axis=0)

    print("Using measured LED directions:")
    for f, d, v in zip(files, d_list, L_measured_subset):
        print(f"  {Path(f).name}: D{d} -> {v}")

    # -------------------------------------------------------------------------
    # Variants (ALL variants now respect measured LED azimuth ordering)
    # -------------------------------------------------------------------------

    variants = {
        "basic": lambda: build_light_dirs(angles_deg=G_ANGLES_DEG),
        "tilted": lambda: build_light_dirs_tilted(
            angles_deg=G_ANGLES_DEG,
            z_tilt=1.4,
            cam_tilt_deg=(18.0, 0.0, 0.0),
        ),
        "point": lambda: build_light_dirs_point(angles_deg=G_ANGLES_DEG),
        "measured": lambda: L6_measured,
    }

    normals_by_variant = {}
    albedo_by_variant = {}

    for name, builder in variants.items():
        L_full = builder()  # (6,3)

        # Subset lights to match input images
        L = np.stack([L_full[g] for g in g_idx_list], axis=0)

        albedo_v, n_v = solve_photometric_stereo(I_cal, L, mask)
        n_uniform_albedo_v = solve_photometric_stereo_uniform_albedo(I_cal, L, mask)

        save_normals_rgb(n_v, str(Path(norm_dir) / f"normals_{name}.png"))
        save_normals_rgb(
            n_uniform_albedo_v,
            str(Path(norm_dir) / f"normals_uniform_albedo_{name}.png")
        )

        normals_by_variant[name] = n_v
        albedo_by_variant[name] = albedo_v

    # --- Choose one variant (measured subset) to produce the rest of the outputs ---
    n = normals_by_variant["tilted"]
    albedo = albedo_by_variant["tilted"]

    # Depth (optionally smooth normals first)
    n_smooth = cv2.bilateralFilter(n.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3)
    z = normals_to_depth(n_smooth, mask)

    # Save other outputs
    save_image(normalize_uint8(albedo), str(Path(albedo_dir) / "albedo.png"))
    save_image(normalize_uint8(np.nan_to_num(z, nan=0.0)), str(Path(depth_dir) / "depth_100.png"))
    save_float_array(z, str(Path(depth_dir) / "depth_tilted.npy"), format="npy")
    save_float_array(z, str(Path(depth_dir) / "depth.pfm"), format="pfm")
    save_depth_plot(z, mask, str(Path(depth_dir) / "depth_3d_tilted.png"))

    # If you compute shadows later and need the light directions used:
    L_measured = L_measured_subset

    save_shadow_maps(n, L_measured, mask, shadow_dir)

    # Mask & composite
    save_image(mask * 255, str(Path(mask_dir) / "mask.png"))
    save_image(normalize_uint8(Imax), str(Path(composite_dir) / "composite_max.png"))

    # Summary
    print("Processed files (first 6):")
    for f in files[:6]:
        print(f" - {f}")
    print(f"Wrote outputs to: {Path(output_dir).resolve()}")

    # Example: after z_rel has been computed from normals
    z_rel = normals_to_depth(n, mask, pixel_size=Config.DEFAULT_PIXEL_SIZE)

    depth_path = Path(depth_dir) / "depth.npy"
    np.save(depth_path, z_rel.astype(np.float32))

    # Calibrate depth with SLI data ---
    # sli_csv_path = Config.resolve_single_csv(Config.DEFAULT_SLI_CSV_GLOB_100mm)
    if sli_csv_glob:
        sli_csv_path = Config.resolve_single_csv(sli_csv_glob)
    else:
        sli_csv_path = None

    cal_depth_path = Path(depth_dir) / "cal_depth.npy"

    if sli_csv_path and sli_csv_path.exists():
        calibrate_depth_with_sli(
            depth_npy_path=str(depth_path),
            sli_csv_path=str(sli_csv_path),
            output_path=str(cal_depth_path),
            use_least_squares=True,
            mask_npy_path=None,
        )
    else:
        print(f"[CALIBRATION] SLI CSV not found for glob: {sli_csv_glob}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photometric Stereo Pipeline")
    args = parser.parse_args()

    main(
        input_glob_or_folder=Config.DEFAULT_INPUT_GLOB_40mm,
        sli_csv_glob=Config.DEFAULT_SLI_CSV_GLOB_40mm,
        working_height=40.0,
        output_dir=Config.DEFAULT_OUTPUT_DIR,
        albedo_dir=Config.OUTPUT_DIR_ALBEDO,
        composite_dir=Config.OUTPUT_DIR_COMPOSITES,
        depth_dir=Config.OUTPUT_DIR_DEPTH,
        mask_dir=Config.OUTPUT_DIR_MASKS,
        norm_dir=Config.OUTPUT_DIR_NORMALIZATION,
        shadow_dir=Config.OUTPUT_DIR_SHADOWS,
        use_otsu=True,
        mask_quantile=Config.DEFAULT_MASK_QUANTILE,
        use_illumination_correction= True,
    )