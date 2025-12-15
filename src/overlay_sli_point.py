from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image, ImageDraw

from .config import Config


def resolve_single_file(pattern: str, kind: str) -> Path:
    """
    Resolve a glob pattern to a single file.

    - pattern: glob pattern (can be absolute)
    - kind: human-readable description ("image", "SLI CSV", ...)
    """
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {kind} found for pattern: {pattern}")
    if len(matches) > 1:
        print(
            f"[overlay_sli_point] Warning: multiple {kind} files for pattern\n"
            f"  {pattern}\n"
            f"  Using: {matches[0]}"
        )
    return Path(matches[0])


# ------------------------------
# CSV loader
# ------------------------------
def load_sli_csv(csv_path: Path):
    """
    Load SLI CSV with columns u_px, v_px and Z/Z_um.

    Returns:
        u, v, Z  (numpy arrays of float)
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Expect u_px / v_px columns
    if "u_px" not in data.dtype.names or "v_px" not in data.dtype.names:
        raise ValueError(
            f"CSV {csv_path} must contain 'u_px' and 'v_px' columns; "
            f"found: {data.dtype.names}"
        )

    u = np.array(data["u_px"], dtype=float)
    v = np.array(data["v_px"], dtype=float)

    # Accept either Z_um or Z
    names = data.dtype.names
    if "Z_um" in names:
        Z = np.array(data["Z_um"], dtype=float)
    elif "Z" in names:
        Z = np.array(data["Z"], dtype=float)
    else:
        raise ValueError(
            f"CSV {csv_path} must contain either 'Z_um' or 'Z' column; "
            f"found: {names}"
        )

    return u, v, Z


# ------------------------------
# Main function
# ------------------------------
def overlay_sli_points(
    image_glob: str | None = None,
    csv_glob: str | None = None,
):
    """
    Overlay SLI points on the first image matching image_glob and
    use the first CSV matching csv_glob.

    Called from src.main.main() with defaults from Config.
    """
    #IMAGE_GLOB = image_glob or Config.DEFAULT_INPUT_GLOB_40mm
    #CSV_GLOB = csv_glob or Config.DEFAULT_SLI_CSV_GLOB_40mm

    IMAGE_GLOB = image_glob or Config.DEFAULT_INPUT_GLOB_100mm
    CSV_GLOB = csv_glob or Config.DEFAULT_SLI_CSV_GLOB_100mm

    PIXEL_ORIGIN = "matlab"  # options: "matlab" (1-based), "python" (0-based)
    POINT_RADIUS = 4

    # Resolve glob patterns to actual files
    img_path = resolve_single_file(IMAGE_GLOB, "input image")
    csv_path = resolve_single_file(CSV_GLOB, "SLI CSV")

    print(f"[overlay_sli_point] Using image: {img_path}")
    print(f"[overlay_sli_point] Using CSV:   {csv_path}")

    # Load image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # Load CSV
    u_raw, v_raw, Z = load_sli_csv(csv_path)

    # Convert MATLAB → Python coordinates
    if PIXEL_ORIGIN == "matlab":
        u = u_raw - 1
        v = v_raw - 1
    else:
        u = u_raw.copy()
        v = v_raw.copy()

    # Clamp to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # ------------------------------
    # Compute Z → colors
    # ------------------------------
    z_min, z_max = float(np.min(Z)), float(np.max(Z))
    print(f"Z range: {z_min} → {z_max}")

    z_close = z_min + (z_max - z_min) / 3.0
    z_mid = z_min + 2 * (z_max - z_min) / 3.0

    colors = []
    for zz in Z:
        if zz <= z_close:
            colors.append("red")
        elif zz <= z_mid:
            colors.append("orange")
        else:
            colors.append("blue")

    # ------------------------------
    # Non-interactive PIL overlay
    # ------------------------------
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    for uu, vv, color in zip(u, v, colors):
        bbox = [
            uu - POINT_RADIUS,
            vv - POINT_RADIUS,
            uu + POINT_RADIUS,
            vv + POINT_RADIUS,
        ]
        draw.ellipse(bbox, fill=color, outline="black")

    # Output directory: PROJECT_ROOT/Output/SLI_Overlay
    base_output = Path(Config.DEFAULT_OUTPUT_DIR)
    out_dir = base_output / "SLI_Overlay"
    Config.ensure_dir(str(out_dir))

    out_path = out_dir / f"{img_path.stem}_sli_overlay.png"
    overlay.save(out_path)
    print(f"[overlay_sli_point] Saved overlay to: {out_path}")


# Optional: allow running this module directly for debugging
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay SLI points on an image")
    parser.add_argument(
        "--image-glob",
        #default=Config.DEFAULT_INPUT_GLOB_40mm,
        default=Config.DEFAULT_INPUT_GLOB_100mm,
        help="Glob for input image(s); first match is used",
    )
    parser.add_argument(
        "--csv-glob",
        #default=Config.DEFAULT_SLI_CSV_GLOB_40mm,
        default=Config.DEFAULT_INPUT_GLOB_100mm,
        help="Glob for SLI CSV file; first match is used",
    )
    args = parser.parse_args()
    overlay_sli_points(args.image_glob, args.csv_glob)
