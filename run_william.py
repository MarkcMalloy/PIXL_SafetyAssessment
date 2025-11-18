from pathlib import Path
import numpy as np
import cv2

from src.config import Config
from src.image_io import load_pngs, save_image, save_float_array
from src.preprocessing import otsu_on_max, quantile_mask, normalize_uint8
from src.photometric_stereo import (
    build_light_dirs,
    build_light_dirs_tilted,
    build_light_dirs_point,
    build_light_dirs_william,
    solve_photometric_stereo,
)
from src.depth_estimation import normals_to_depth
from src.visualization import save_normals_rgb, save_depth_plot

INPUT_GLOB = "PIXL_Images/TestData"     # choose your input images
# INPUT_GLOB = "PIXL_Images/CalData_william/NoObstacle"
# INPUT_GLOB = "PIXL_Images/CalData_william/WithObstacle"
OUTPUT_DIR = "Output_william"           # choose output directory

USE_OTSU = True                   # True → Otsu mask, False → quantile mask
MASK_QUANTILE = 0.95              # Only used if USE_OTSU = False

# START
print(f"Loading input images: {INPUT_GLOB}")

# Make sure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(OUTPUT_DIR)

# Load images
I, files = load_pngs(INPUT_GLOB)
print(f"Loaded {len(files)} images.")

# Mask
if USE_OTSU:
    mask, Imax = otsu_on_max(I)
else:
    Imax = I.max(axis=-1)
    mask = quantile_mask(I, MASK_QUANTILE)

save_image(mask * 255, str(OUTPUT_DIR / "mask.png"))

# Per-light gain calibration
mean_intensity = [I[..., i].mean() for i in range(I.shape[-1])]
s = np.array(mean_intensity) / np.mean(mean_intensity)  
I_cal = I / s[None, None, :]

# Build light locations
angles = [64, 113, 162, 211, 259, 308] # FLI Degrees starting from x=0, y=-r going clockwise
angles = [(270 - a) % 360 for a in angles]  # Convert to standard math degrees starting from x=+r, y=0 going counter-clockwise
r = 16.5  # mm, radius of LED ring
d = 56 # mm, distance from target plane to MCC pinhole

L = build_light_dirs_william(angles, r, d)

# Solve photometric stereo
albedo, normals = solve_photometric_stereo(I_cal, L, mask)
save_image(normalize_uint8(albedo), str(OUTPUT_DIR / "albedo.png"))
save_image(normalize_uint8(normals), str(OUTPUT_DIR / "normals.png"))

# Estimate depth from normals
depth = normals_to_depth(normals, mask)
save_image(normalize_uint8(depth), str(OUTPUT_DIR / "depth.png"))
save_depth_plot(depth, mask, str(OUTPUT_DIR / "depth_3d.png"))

print("Processing complete.")

# from src.main import main  # absolute import now
# from pathlib import Path

# # Set paths
# # input_folder = Path("PIXL_Images/TestData")
# input_folder = Path("PIXL_Images/CalData_william/NoObstacle")
# # input_folder = Path("PIXL_Images/CalData_william/WithObstacle")
# output_folder = Path("Output_william")

# main(
#     input_glob_or_folder=input_folder,
#     output_dir=output_folder,
#     albedo_dir=output_folder / "albedo",
#     composite_dir=output_folder / "composites",
#     depth_dir=output_folder / "depth",
#     mask_dir=output_folder / "masks",
#     norm_dir=output_folder / "normals",
#     shadow_dir=output_folder / "shadows",
#     use_otsu=True,
#     mask_quantile=0.95
# )

# print("Processing complete.")