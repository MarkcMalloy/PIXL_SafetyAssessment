import numpy as np
import cv2
from .image_io import save_image
from src.config import Config
from pathlib import Path


### The purpose of this script is to handle visualization and data output generation.

def save_normals_rgb(n: np.ndarray, path: str):
    """Map normals to RGB for visualization."""
    nn = (n + 1.0) * 0.5
    img = np.clip(nn * 255.0, 0, 255).astype(np.uint8)
    save_image(img, path, convert_bgr=True)

def save_shadow_maps(n: np.ndarray, L: np.ndarray, mask: np.ndarray, out_dir: str):
    """Save per-light shadow maps."""
    Config.ensure_dir(out_dir)
    for i, s in enumerate(L):
        ndotl = np.sum(n * s, axis=-1)
        shadow = (ndotl <= 0).astype(np.uint8) * 255
        shadow[mask == 0] = 0
        save_image(shadow, str(Path(out_dir) / f"shadow_{i}.png"))