# src/segmentation.py
"""
Segment foreground rock/obstacles from background terrain.

Usage (from project root):
    python -m src.segmentation \
        --input "Output/Composites/*.png"

This will create:
    Output/Segmentation/<stem>_foreground.png
    Output/Segmentation/<stem>_background.png
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from config import Config

out_dir = Path(Config.OUTPUT_DIR_SEGMENTATION)


def collect_image_paths(input_glob_or_folder: str) -> list[Path]:
    """
    Accept:
      - a folder path
      - a single file path
      - a glob pattern (relative or absolute)
    and return a sorted list of image paths.
    """
    p = Path(input_glob_or_folder)

    # Directory -> take all PNGs
    if p.exists() and p.is_dir():
        return sorted(p.glob("*.png"))

    # Single file
    if p.exists() and p.is_file():
        return [p]

    # Glob pattern
    if p.is_absolute():
        return sorted(p.parent.glob(p.name))
    else:
        return sorted(Path(".").glob(input_glob_or_folder))


def segment_foreground_enhanced(
        gray: np.ndarray,
        use_adaptive: bool = True,
        edge_weight: float = 0.4,
) -> np.ndarray:
    """
    Enhanced segmentation combining:
    1. Adaptive histogram equalization
    2. Multi-scale edge detection
    3. Texture analysis
    4. GrabCut refinement

    Returns mask_fg: uint8, 1 = foreground (rock), 0 = background (terrain).
    """
    h, w = gray.shape

    # ---------- 1) Preprocessing with CLAHE ----------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ---------- 2) Multi-scale edge detection ----------
    # Sobel edges at different scales
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Canny edges for fine details
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 100)

    # Combine edge information
    edges_combined = cv2.addWeighted(sobel_mag, 0.6, canny, 0.4, 0)

    # ---------- 3) Texture-based segmentation ----------
    # Local standard deviation as texture measure
    kernel_size = 15
    local_mean = cv2.blur(enhanced.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((enhanced.astype(np.float32)) ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    local_std = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ---------- 4) Combine features ----------
    # Weight: intensity + edges + texture
    feature_map = cv2.addWeighted(enhanced, 0.4, edges_combined, edge_weight, 0)
    feature_map = cv2.addWeighted(feature_map, 0.7, local_std, 0.3, 0)

    # ---------- 5) Initial segmentation with adaptive threshold ----------
    if use_adaptive:
        # Adaptive threshold works better for varying illumination
        thresh = cv2.adaptiveThreshold(
            feature_map, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=-5
        )
    else:
        # Otsu's method as fallback
        _, thresh = cv2.threshold(feature_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---------- 6) Morphological refinement ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Close small holes in foreground
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---------- 7) Find largest connected component (the main rock) ----------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels > 1:
        # Find largest component (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_fg = (labels == largest_label).astype(np.uint8)
    else:
        mask_fg = (thresh > 0).astype(np.uint8)

    # ---------- 8) Optional: GrabCut refinement ----------
    # This helps refine boundaries
    if np.sum(mask_fg) > 0.01 * h * w:  # Only if we have a reasonable foreground
        # Create GrabCut mask
        grabcut_mask = np.zeros(gray.shape, dtype=np.uint8)
        grabcut_mask[mask_fg == 1] = cv2.GC_PR_FGD  # Probably foreground
        grabcut_mask[mask_fg == 0] = cv2.GC_PR_BGD  # Probably background

        # Make sure edges are definite
        kernel_sure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        sure_fg = cv2.erode(mask_fg, kernel_sure, iterations=2)
        sure_bg = cv2.dilate(1 - mask_fg, kernel_sure, iterations=2)

        grabcut_mask[sure_fg == 1] = cv2.GC_FGD
        grabcut_mask[sure_bg == 1] = cv2.GC_BGD

        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Convert to BGR for GrabCut
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.grabCut(gray_bgr, grabcut_mask, None, bgd_model, fgd_model,
                        iterCount=3, mode=cv2.GC_INIT_WITH_MASK)

            mask_fg = np.where((grabcut_mask == cv2.GC_FGD) |
                               (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except:
            # If GrabCut fails, keep original mask
            pass

    # ---------- 9) Final cleanup ----------
    # Smooth the boundary slightly
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Small erosion to ensure clean separation
    mask_fg = cv2.erode(mask_fg, np.ones((3, 3), np.uint8), iterations=1)

    return mask_fg


def segment_foreground_watershed(
    gray: np.ndarray,
    bg_frac: float = 0.35,
    fg_frac: float = 0.60,
) -> np.ndarray:
    """
    Edge-aware 2-class segmentation using watershed.

    gray : 2D grayscale image
    bg_frac : fraction of image height used as sure-background (top part)
    fg_frac : fraction of image height where sure-foreground starts (bottom part)

    Returns:
        mask_fg (uint8): 1 = foreground (rock), 0 = background (terrain)
    """
    h, w = gray.shape

    # Slight blur to reduce noise, keep edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Watershed needs 3-channel image
    img_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # ----- Build markers in uint8 first -----
    # 0 = unknown, 1 = background, 2 = foreground
    markers8 = np.zeros((h, w), np.uint8)

    bg_end = int(bg_frac * h)
    fg_start = int(fg_frac * h)

    markers8[:bg_end, :] = 1   # background label
    markers8[fg_start:, :] = 2 # foreground label

    # Erode seeds slightly for safety (uint8 is supported)
    kernel = np.ones((3, 3), np.uint8)
    markers8 = cv2.erode(markers8, kernel, iterations=1)

    # Convert to int32 for watershed
    markers = markers8.astype(np.int32)

    # ----- Watershed -----
    cv2.watershed(img_bgr, markers)
    # markers == -1 are watershed boundaries

    mask_fg = (markers == 2).astype("uint8")

    # Clean up mask: close small gaps, remove noise
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask_fg

def save_segmented_images(
        img_bgr: np.ndarray,
        mask_fg: np.ndarray,
        stem: str,
) -> None:
    """
    Save two images:
      - *_foreground.png  (rock)
      - *_background.png  (terrain)
    """
    if mask_fg.shape != img_bgr.shape[:2]:
        raise ValueError(
            f"Mask shape {mask_fg.shape} != image shape {img_bgr.shape[:2]}"
        )

    fg = img_bgr.copy()
    fg[mask_fg == 0] = 0

    bg = img_bgr.copy()
    bg[mask_fg == 1] = 0

    # Also save the mask for inspection
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask_fg * 255)
    cv2.imwrite(str(out_dir / f"{stem}_foreground.png"), fg)
    cv2.imwrite(str(out_dir / f"{stem}_background.png"), bg)


def run_segmentation(input_glob_or_folder: str, use_enhanced: bool = True):
    """
    Run segmentation on images.

    Args:
        input_glob_or_folder: Path to images
        use_enhanced: If True, use enhanced segmentation; else use watershed
    """
    img_paths = collect_image_paths(input_glob_or_folder)
    if not img_paths:
        print(f"[Segmentation] No images found for: {input_glob_or_folder}")
        return

    print(f"[Segmentation] Found {len(img_paths)} images")
    print(f"[Segmentation] Output dir: {out_dir}")
    print(f"[Segmentation] Method: {'Enhanced' if use_enhanced else 'Watershed'}")

    for path in img_paths:
        print(f"  - segmenting {path}")
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"    WARNING: could not read {path}, skipping.")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if use_enhanced:
            mask_fg = segment_foreground_enhanced(gray)
        else:
            mask_fg = segment_foreground_watershed(gray)

        save_segmented_images(img_bgr, mask_fg, path.stem)


def main_cli() -> None:
    run_segmentation(
        input_glob_or_folder=Config.OUTPUT_DIR_COMPOSITES,
        use_enhanced=False  # Use enhanced method by default
    )


if __name__ == "__main__":
    main_cli()