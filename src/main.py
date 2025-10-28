import argparse
from pathlib import Path
import numpy as np
import cv2  # needed for bilateralFilter

from .config import Config
from .image_io import load_pngs, save_image, save_float_array
from .preprocessing import otsu_on_max, quantile_mask, normalize_uint8
from .photometric_stereo import (
    build_light_dirs,            # basic ring light model
    build_light_dirs_tilted,     # ring + camera tilt
    build_light_dirs_point,      # point-light (tilt + camera offset)
    solve_photometric_stereo,
)
from .depth_estimation import normals_to_depth
from .visualization import save_normals_rgb, save_shadow_maps
from .normals_to_mesh import depth_to_mesh


def main(
        input_glob_or_folder: str = Config.DEFAULT_INPUT_GLOB,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR,
        albedo_dir: str = Config.OUTPUT_DIR_ALBEDO,
        composite_dir: str = Config.OUTPUT_DIR_COMPOSITES,
        depth_dir: str = Config.OUTPUT_DIR_DEPTH,
        mask_dir: str = Config.OUTPUT_DIR_MASKS,
        norm_dir: str = Config.OUTPUT_DIR_NORMALIZATION,
        shadow_dir: str = Config.OUTPUT_DIR_SHADOWS,
        mesh_dir: str = Config.OUTPUT_DIR_MESH,
        use_otsu: bool = True,
        mask_quantile: float = Config.DEFAULT_MASK_QUANTILE
):
    # Ensure output directories exist (now includes mesh_dir)
    for d in [output_dir, albedo_dir, composite_dir, depth_dir, mask_dir, norm_dir, shadow_dir, mesh_dir]:
        Config.ensure_dir(d)

    print(f"Input glob: {input_glob_or_folder}")
    print(f"Output directory: {output_dir}")

    # Load images
    I, files = load_pngs(input_glob_or_folder)

    # Mask either otsu or quantile (otsu gives best results?)
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
    variants = {
        "basic":  build_light_dirs,
        "tilted": build_light_dirs_tilted,
        "point":  build_light_dirs_point,
    }

    normals_by_variant = {}
    albedo_by_variant = {}

    for name, builder in variants.items():
        L = builder()
        albedo_v, n_v = solve_photometric_stereo(I_cal, L, mask)
        save_normals_rgb(n_v, str(Path(norm_dir) / f"normals_{name}.png"))
        normals_by_variant[name] = n_v
        albedo_by_variant[name] = albedo_v

    # --- Produce depths and meshes for ALL variants ---
    for name, n_v in normals_by_variant.items():
        # (optional) smooth normals before integration
        n_smooth = cv2.bilateralFilter(n_v.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3)
        z_v = normals_to_depth(n_smooth, mask)  # HxW float32, NaN outside mask

        # Save per-variant depth
        depth_png_path = Path(depth_dir) / f"depth_{name}.png"
        depth_npy_path = Path(depth_dir) / f"depth_{name}.npy"
        save_image(normalize_uint8(np.nan_to_num(z_v, nan=0.0)), str(depth_png_path))
        save_float_array(z_v, str(depth_npy_path), format="npy")

        # Mesh output subdir per variant
        mesh_out = Path(mesh_dir) / name
        Config.ensure_dir(mesh_out)

        # --- Adaptive Poisson depth selection ---
        poisson_depth = 6 if name == "point" else 7
        print(f"[INFO] Using Poisson depth={poisson_depth} for variant '{name}'")

        # Generate 3D mesh
        depth_to_mesh(
            depth_path=str(depth_npy_path),
            output_dir=str(mesh_out),
            scale=1.0,
            mask_path=str(Path(mask_dir) / "mask.png"),
            poisson_depth=poisson_depth
        )

    # For legacy outputs, keep 'point' as the primary variant
    albedo = albedo_by_variant["point"]
    n = normals_by_variant["point"]

    # Shadows from point lights
    L_point = build_light_dirs_point()
    save_shadow_maps(n, L_point, mask, shadow_dir)

    # Albedo, composite, mask
    save_image(normalize_uint8(albedo), str(Path(albedo_dir) / "albedo.png"))
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
    parser.add_argument("--mesh-dir", default=Config.OUTPUT_DIR_MESH, help="Meshes output directory")  # <-- added
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
        mesh_dir=args.mesh_dir,  # <-- now defined
        use_otsu=args.use_otsu,
        mask_quantile=args.mask_quantile
    )
