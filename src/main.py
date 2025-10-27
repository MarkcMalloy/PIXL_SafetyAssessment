import argparse
from pathlib import Path
import numpy as np
from .config import Config
from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import otsu_on_max, quantile_mask, normalize_uint8
from .photometric_stereo import build_light_dirs, solve_photometric_stereo
from .depth_estimation import normals_to_depth
from .visualization import save_normals_rgb, save_shadow_maps

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
    # Ensure all output directories exist
    # TODO: Remove ensure_dir in future since it will be redundant. For now it is nice to have this safety feature
    Config.ensure_dir(output_dir)
    Config.ensure_dir(albedo_dir)
    Config.ensure_dir(composite_dir)
    Config.ensure_dir(depth_dir)
    Config.ensure_dir(mask_dir)
    Config.ensure_dir(norm_dir)
    Config.ensure_dir(shadow_dir)
    print(f"Input glob: {input_glob_or_folder}")
    print(f"Output directory: {output_dir}")
    # Load images
    I, files = load_pngs(input_glob_or_folder)

    # Generate mask
    if use_otsu:
        mask, Imax = otsu_on_max(I)
    else:
        Imax = I.max(axis=-1)
        mask = quantile_mask(I, mask_quantile)

    # Compute light directions
    L = build_light_dirs()

    # Solve photometric stereo
    albedo, n = solve_photometric_stereo(I, L, mask)

    # Estimate depth
    z = normals_to_depth(n, mask)

    # Save outputs
    save_image(normalize_uint8(albedo), str(Path(albedo_dir) / "albedo.png"))
    save_normals_rgb(n, str(Path(norm_dir) / "normals.png"))
    save_image(normalize_uint8(np.nan_to_num(z, nan=0.0)), str(Path(depth_dir) / "depth.png"))
    save_float_array(z, str(Path(depth_dir) / "depth.npy"), format="npy")
    save_float_array(z, str(Path(depth_dir) / "depth.pfm"), format="pfm")
    save_shadow_maps(n, L, mask, shadow_dir)  # No need for Path here, handled in save_shadow_maps
    save_image(mask * 255, str(Path(mask_dir) / "mask.png"))
    save_image(normalize_uint8(Imax), str(Path(composite_dir) / "composite_max.png"))

    # Print summary
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