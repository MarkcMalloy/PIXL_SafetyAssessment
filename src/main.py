import argparse
from pathlib import Path
import numpy as np
from src.config import Config
from src.image_io import load_pngs, save_image, save_float_array
from src.preprocessing import otsu_on_max, quantile_mask, normalize_uint8
from src.photometric_stereo import build_light_dirs, solve_photometric_stereo
from src.depth_estimation import normals_to_depth
from src.visualization import save_normals_rgb, save_shadow_maps

### This is the main pipeline for creating a stereo image from shade data
### It works by calling the other modules in /src
def main(
        input_glob_or_folder: str = Config.DEFAULT_INPUT_GLOB,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR,
        use_otsu: bool = True,
        mask_quantile: float = Config.DEFAULT_MASK_QUANTILE
):
    Config.ensure_dir(output_dir)
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
    save_image(normalize_uint8(albedo), str(Path(output_dir) / "albedo.png"))
    save_normals_rgb(n, str(Path(output_dir) / "normals.png"))
    save_image(normalize_uint8(np.nan_to_num(z, nan=0.0)), str(Path(output_dir) / "depth.png"))
    save_float_array(z, str(Path(output_dir) / "depth.npy"), format="npy")
    save_float_array(z, str(Path(output_dir) / "depth.pfm"), format="pfm")
    save_shadow_maps(n, L, mask, str(Path(output_dir) / "Shadows"))
    save_image(mask * 255, str(Path(output_dir) / "mask.png"))
    save_image(normalize_uint8(Imax), str(Path(output_dir) / "composite_max.png"))

    # Print summary
    print("Processed files (first 6):")
    for f in files[:6]:
        print(f" - {f}")
    print(f"Wrote outputs to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photometric Stereo Pipeline")
    parser.add_argument("--input", default=Config.DEFAULT_INPUT_GLOB, help="Input glob or folder")
    parser.add_argument("--output", default=Config.DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-otsu", action="store_false", dest="use_otsu", help="Use quantile mask instead of Otsu")
    parser.add_argument("--mask-quantile", type=float, default=Config.DEFAULT_MASK_QUANTILE, help="Quantile for mask")
    args = parser.parse_args()
    main(args.input, args.output, args.use_otsu, args.mask_quantile)