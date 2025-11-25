import argparse
from pathlib import Path
import numpy as np
from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import normalize_uint8
from .sfs_horn import shape_from_shading
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
    if I.max() > 1.0 + 1e-3:   # likely uint8 0..255
        I /= 255.0

    # ------------------------------------------------------------------ #
    # 2. Build a single brightness image E(x,y) for SFS
    #    (Here: simple average over the 6 LED images)
    # ------------------------------------------------------------------ #
    E = I.mean(axis=-1)  # shape (H, W)

    # Save the composite used as SFS input (for debugging/visualization)
    save_image(
        normalize_uint8(E),
        str(Path(output_dir) / "sfs_input_composite.png"),
    )

    # ------------------------------------------------------------------ #
    # 3. Shape from shading (Horn-style, rotationally symmetric)
    # ------------------------------------------------------------------ #
    # You can tweak these hyperparameters later
    z, p, q = shape_from_shading(
        E,
        rho=1.0,
        n_iters=200,  # enough iterations
        step=0.02,  # smaller step for stability
        lam_smooth=2.0,  # MUCH stronger smoothness
        verbose=True,  # optional: see loss evolution
    )

    # ------------------------------------------------------------------ #
    # 4. Save outputs (all in output_dir)
    # ------------------------------------------------------------------ #
    depth_base = Path(output_dir) / "sfs_depth"

    # Depth (relative units)
    save_image(
        normalize_uint8(np.nan_to_num(z, nan=0.0)),
        str(depth_base.with_suffix(".png")),
    )
    save_float_array(z, str(depth_base.with_suffix(".npy")), format="npy")
    save_float_array(z, str(depth_base.with_suffix(".pfm")), format="pfm")

    # Optional: save slope fields p = dz/dx, q = dz/dy
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
