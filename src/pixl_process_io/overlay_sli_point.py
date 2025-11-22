from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw


def load_sli_csv(csv_path: str | Path):
    """
    Load SLI CSV with columns at least: u,v,X,Y,Z (header row required).

    Returns:
        u_raw, v_raw : 1D numpy arrays with image-plane coordinates
        X, Y, Z      : 1D numpy arrays (e.g. microns in your case)
    """
    csv_path = Path(csv_path)

    # Use genfromtxt so we can handle headers by name
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Expect at least columns 'u' and 'v'
    if "u" not in data.dtype.names or "v" not in data.dtype.names:
        raise ValueError(
            f"{csv_path} must have at least 'u' and 'v' columns in the header. "
            f"Found columns: {data.dtype.names}"
        )

    u_raw = np.array(data["u"], dtype=float)
    v_raw = np.array(data["v"], dtype=float)

    # Optional X,Y,Z
    X = np.array(data[data.dtype.names[2]], dtype=float) if len(data.dtype.names) > 2 else None
    Y = np.array(data[data.dtype.names[3]], dtype=float) if len(data.dtype.names) > 3 else None
    Z = np.array(data[data.dtype.names[4]], dtype=float) if len(data.dtype.names) > 4 else None

    return u_raw, v_raw, X, Y, Z


def overlay_all_points(
    image_path: str | Path,
    csv_path: str | Path,
    out_path: str | Path | None = None,
    pixel_origin: str = "matlab",
    point_radius: int = 1.4,
):
    """
    Overlay SLI points from CSV onto an image.

    Args:
        image_path: path to the base image (e.g. normals image).
        csv_path: CSV file with at least columns u,v.
        out_path: where to save the overlay PNG. If None, uses
                  <image_stem>_sli_overlay.png in the current directory.
        pixel_origin: "matlab" for 1-based (u,v) coordinates, "python" for 0-based.
        point_radius: radius (in pixels) of the drawn points.
    """
    image_path = Path(image_path)
    csv_path = Path(csv_path)

    # Load base image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Load SLI points
    u_raw, v_raw, X, Y, Z = load_sli_csv(csv_path)

    # Convert to 0-based pixel coordinates if needed
    if pixel_origin.lower() == "matlab":
        u = u_raw - 1.0
        v = v_raw - 1.0
    else:
        u = u_raw.copy()
        v = v_raw.copy()

    # Clamp to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1) +133

    # Prepare drawing
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # Draw each point as a small circle
    for uu, vv in zip(u, v):
        x = float(uu)
        y = float(vv)
        bbox = [
            x - point_radius,
            y - point_radius,
            x + point_radius,
            y + point_radius,
        ]
        # red points with a black outline
        draw.ellipse(bbox, outline="black", fill="red")

    # Decide output path
    if out_path is None:
        out_path = Path.cwd() / f"{image_path.stem}_sli_overlay.png"
    else:
        out_path = Path(out_path)

    overlay.save(out_path)
    print(f"Saved overlay with all SLI points to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay SLI points (u,v) from CSV onto an image."
    )
    #parser.add_argument(
        #"-i", "--image",
       # type=str,
      #  default="40mm_2.png", # Use to calibrate Y position with light point
     #   help="Image file in the current directory (default: normals_point.png)",
    #)
    parser.add_argument(
        "-i", "--image",
        type=str,
        default="normals_point.png",
        help="Image file in the current directory (default: normals_point.png)",
    )
    parser.add_argument(
        "-c", "--csv",
        type=str,
        default=None,
        help="CSV file with SLI points in the current directory "
             "(default: first *SLI_points*.csv found)",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="matlab",
        choices=["matlab", "python"],
        help="Coordinate origin convention for u,v (default: matlab = 1-based).",
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default=None,
        help="Output PNG filename (default: <image_stem>_sli_overlay.png in pwd)",
    )

    args = parser.parse_args()

    # Resolve image path in current working directory
    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Resolve CSV path: either explicit or auto-detect *SLI_points*.csv
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        candidates = sorted(Path.cwd().glob("*SLI_points*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "No CSV provided and no *SLI_points*.csv found in current directory."
            )
        csv_path = candidates[0]
        print(f"Using CSV: {csv_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    overlay_all_points(
        image_path=img_path,
        csv_path=csv_path,
        out_path=args.out,
        pixel_origin=args.origin,
    )


if __name__ == "__main__":
    main()
