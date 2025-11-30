import argparse
from pathlib import Path
import numpy as np

from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import normalize_uint8, otsu_on_max
from .sfs_horn import (
    shape_from_shading_multilight_const_albedo,
    build_led_dirs_measured,
)
from .config import Config
from .visualization import save_depth_plot


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

    # Save depth
    depth_base = Path(output_dir) / "sfs_depth"
    save_image(
        normalize_uint8(np.nan_to_num(z, nan=0.0)),
        str(depth_base.with_suffix(".png")),
    )
    save_depth_plot(normalize_uint8(np.nan_to_num(z, nan=0.0)), str("Output/sfs_depth_3d.png"))

    save_float_array(z, str(depth_base.with_suffix(".npy")), format="npy")
    save_float_array(z, str(depth_base.with_suffix(".pfm")), format="pfm")

    # Save slope fields
    save_float_array(p, str(Path(output_dir) / "sfs_p.npy"), format="npy")
    save_float_array(q, str(Path(output_dir) / "sfs_q.npy"), format="npy")

    print("Shape-from-shading finished.")
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
