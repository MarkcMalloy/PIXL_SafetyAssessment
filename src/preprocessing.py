import numpy as np
import cv2
from .config import Config

### The purpose of this script is to handle image preprocessing, including normalization and mask generation.
###
def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an image to [0,255] uint8 for visualization (ignores NaNs)."""
    m = np.isfinite(img)
    if not np.any(m):
        return np.zeros_like(img, dtype=np.uint8)
    a, b = img[m].min(), img[m].max()
    if b <= a + 1e-12:
        out = np.zeros_like(img, dtype=np.uint8)
        out[m] = 0
        return out
    out = np.zeros_like(img, dtype=np.float32)
    out[m] = (img[m] - a) / (b - a)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def otsu_on_max(I: np.ndarray, morph_open_ksize: int = Config.DEFAULT_MORPH_OPEN_KSIZE) -> tuple[np.ndarray, np.ndarray]:
    """Foreground mask from Otsu thresholding on max-intensity composite."""
    Imax = I.max(axis=-1)
    I8 = normalize_uint8(Imax)
    thr, _ = cv2.threshold(I8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # This is pretty important for stereo determination
    mask = (I8 >= int(thr * 0.50)).astype(np.uint8)  # Multiplying the thr value between 0-0.99 determines how aggressive we handle shadows
    if morph_open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask, Imax

def quantile_mask(I: np.ndarray, quantile: float = Config.DEFAULT_MASK_QUANTILE) -> np.ndarray:
    """Generate mask using quantile thresholding."""
    Imax = I.max(axis=-1)
    t = np.quantile(Imax, quantile)
    return (Imax > t).astype(np.uint8)