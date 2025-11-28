import numpy as np
from src.config import Config
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

def calibrate_depth_with_sli(
    z_sfs: np.ndarray,
    mask: np.ndarray,
    sli_array: np.ndarray,
    u_col: int = 0,
    v_col: int = 1,
    Z_col: int = 4,
) -> tuple[np.ndarray, float, float]:
    """
    Calibrate relative depth z_sfs using SLI points.

    Parameters
    ----------
    z_sfs : (H,W) float32
        Relative depth from normals_to_depth.
    mask : (H,W) uint8/bool
        Valid mask from your pipeline.
    sli_array : (N,M)
        Raw SLI CSV data as a numeric array.
        Assumed columns: [u_px, v_px, X, Y, Z] by default.
    u_col, v_col, Z_col : int
        Column indices in sli_array for u, v, Z.

    Returns
    -------
    z_cal : (H,W) float32
        Calibrated metric depth.
    a, b : float
        Scale and offset such that Z ≈ a*z_sfs + b
    """
    H, W = z_sfs.shape

    u = sli_array[:, u_col].astype(int)
    v = sli_array[:, v_col].astype(int)
    Z_sli = sli_array[:, Z_col].astype(np.float32)

    # Keep only points inside the image and inside the mask
    valid = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        np.isfinite(z_sfs[v, u]) &
        (mask[v, u] > 0) &
        np.isfinite(Z_sli)
    )
    if not np.any(valid):
        raise RuntimeError("No valid SLI points inside image/mask for depth calibration.")

    u = u[valid]
    v = v[valid]
    Z_sli = Z_sli[valid]
    z_samples = z_sfs[v, u]

    # Optional: filter out points where z_samples is extremely close to mean (very flat),
    # or where SLI depth looks like an outlier. For now, keep it simple.

    # Build least-squares system: [z_i, 1] [a, b]^T ≈ Z_i
    A = np.stack([z_samples, np.ones_like(z_samples)], axis=1)  # (N,2)
    (a, b), *_ = np.linalg.lstsq(A, Z_sli, rcond=None)

    z_cal = a * z_sfs + b
    z_cal[mask == 0] = np.nan

    return z_cal.astype(np.float32), float(a), float(b)