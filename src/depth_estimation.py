import numpy as np
from .config import Config
### The purpose of this script is to handle depth estimation from normals.

def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Integrate gradients p=dz/dx, q=dz/dy into a surface z."""
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2 * np.pi
    wy = np.fft.fftfreq(H) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + WY**2
    denom[0, 0] = 1.0
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    Z = (-1j * WX * P - 1j * WY * Q) / denom
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z)).astype(np.float32)
    z -= np.nanmean(z)
    return z

def normals_to_depth(n: np.ndarray, mask: np.ndarray, pixel_size: float = Config.DEFAULT_PIXEL_SIZE) -> np.ndarray:
    """Convert normals to depth via gradient integration."""
    eps = 1e-8
    nz = np.where(n[..., 2] == 0, eps, n[..., 2])
    p = -n[..., 0] / (nz + eps)
    q = -n[..., 1] / (nz + eps)
    p *= pixel_size
    q *= pixel_size
    m = mask.astype(bool)
    p[~m] = 0.0
    q[~m] = 0.0
    z = frankot_chellappa(p, q)
    z[~m] = np.nan
    return z.astype(np.float32)