from pathlib import Path
import numpy as np
import cv2
from .config import Config

### This script handles image loading and saving operations to isolate the I/O logic from the rest of the program
###
def load_pngs(folder_or_glob: str) -> tuple[np.ndarray, list[str]]:
    """Load 6 grayscale PNGs sorted by name. Returns HxWx6 float32 in [0,1]."""
    p = Path(folder_or_glob)
    if p.is_dir():
        files = sorted([str(f) for f in p.glob("*.png")])
    else:
        files = sorted([str(f) for f in Path().glob(folder_or_glob)])
    if len(files) < Config.NUM_IMAGES:
        raise ValueError(f"Expected at least {Config.NUM_IMAGES} PNGs, found {len(files)} at {folder_or_glob}")
    files = files[:Config.NUM_IMAGES]
    imgs = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Failed to read: {f}")
        imgs.append(im.astype(np.float32) / 255.0)
    return np.stack(imgs, axis=-1), files

def save_image(img: np.ndarray, path: str, convert_bgr: bool = False):
    """Save image to disk, optionally converting RGB to BGR."""
    if convert_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def save_float_array(arr: np.ndarray, path: str, format: str = "npy"):
    """Save float array as .npy or .pfm."""
    if format == "npy":
        np.save(path, np.nan_to_num(arr, nan=0.0).astype(np.float32))
    elif format == "pfm":
        cv2.imwrite(path, np.nan_to_num(arr, nan=0.0).astype(np.float32))
    else:
        raise ValueError(f"Unsupported format: {format}")