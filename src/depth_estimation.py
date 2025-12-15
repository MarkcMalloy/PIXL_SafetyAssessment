from __future__ import annotations
import numpy as np
from pathlib import Path
from .config import Config   # already imported at top in your file
### The purpose of this script is to handle depth estimation from normals.

def _load_sli_csv(csv_path: str):
    """
    Load SLI CSV exported from MATLAB.

    Expected column order:
        u_px, v_px, X_mm, Y_mm, Z_mm

    If you have a header row, set `skiprows=1` in np.loadtxt.
    """
    data = np.loadtxt(csv_path, delimiter=",", comments="#", skiprows=1)

    u_px = data[:, 0]
    v_px = data[:, 1]
    Z_mm = data[:, 4]

    return u_px, v_px, Z_mm


def _build_valid_mask_from_depth(z_rel: np.ndarray) -> np.ndarray:
    """
    Build a mask of 'valid' object/terrain pixels from a relative depth map.

    Idea:
      - Start with all finite pixels.
      - Estimate a background value from the image border.
      - Mark pixels that differ significantly from this background
        as 'object/terrain'.

    This is a heuristic so that SLI points that land in the flat
    background ring of depth.npy are ignored.
    """
    H, W = z_rel.shape
    finite = np.isfinite(z_rel)

    if not finite.any():
        raise RuntimeError("depth.npy contains no finite values.")

    # Collect border pixels
    border_vals = np.concatenate([
        z_rel[0, :],
        z_rel[-1, :],
        z_rel[:, 0],
        z_rel[:, -1],
    ])
    border_vals = border_vals[np.isfinite(border_vals)]

    # If border is unusable, fall back to 'finite' only
    if border_vals.size == 0:
        return finite

    bg = np.median(border_vals)
    mad = np.median(np.abs(border_vals - bg))
    if mad == 0:
        mad = 1e-6

    diff = np.abs(z_rel - bg)
    # Pixels that differ > 3 * MAD from border median are considered object/terrain
    mask = finite & (diff > 3.0 * mad)

    # If that produces almost nothing, fall back to all finite pixels
    if mask.sum() < 100:
        mask = finite

    return mask


def calibrate_depth_with_sli(
    depth_npy_path: str,
    sli_csv_path: str,
    output_path: str | None = None,
    use_least_squares: bool = True,
    mask_npy_path: str | None = None,
):
    """
    Calibrate a relative depth map (depth.npy) using SLI Z values.

    NOTE:
        SLI Z values are provided in microns.
        We convert to millimeters by dividing by 1000.

    depth_npy_path : path to depth.npy (relative z_rel values)
    sli_csv_path   : path to SLI CSV with (u_px, v_px, X, Y, Z_microns)
    output_path    : where to write cal_depth.npy
    use_least_squares : if True, fit a & b using *all* valid SLI points
    mask_npy_path  : optional mask .npy restricting SLI calibration region
    Returns:
        z_abs  : calibrated depth map in *millimeters*
        a, b   : scale and offset such that Z_mm = a * z_rel + b
    """
    depth_npy_path = Path(depth_npy_path)
    z_rel = np.load(depth_npy_path)  # shape (H, W)
    H, W = z_rel.shape

    # ---- Load / build valid mask (object/terrain only) ----
    if mask_npy_path is not None and Path(mask_npy_path).exists():
        valid_mask = np.load(mask_npy_path).astype(bool)
        if valid_mask.shape != z_rel.shape:
            raise RuntimeError("Mask shape does not match depth.npy shape.")
    else:
        valid_mask = _build_valid_mask_from_depth(z_rel)

    # ---- Load SLI points ----
    u_px, v_px, Z_microns = _load_sli_csv(sli_csv_path)

    # Convert microns → millimeters
    Z_abs = Z_microns / 1000.0

    # Convert pixel coords to integer indices
    u_i = np.clip(np.round(u_px).astype(int), 0, W - 1)
    v_i = np.clip(np.round(v_px).astype(int), 0, H - 1)

    # Sample the relative depth at the SLI pixel positions
    z_samples = z_rel[v_i, u_i]

    # Only keep SLI points that land inside the object/terrain part of depth.npy
    inside_depth = valid_mask[v_i, u_i]

    # Finite and inside-object constraint
    valid = np.isfinite(z_samples) & np.isfinite(Z_abs) & inside_depth

    if np.count_nonzero(valid) < 2:
        raise RuntimeError(
            "Not enough valid SLI/depth correspondences for calibration "
            "(need at least 2 inside the object/terrain region)."
        )

    # Filter arrays to only valid correspondences
    z_samples_valid = z_samples[valid]
    Z_abs_valid = Z_abs[valid]
    u_valid = u_px[valid]
    v_valid = v_px[valid]
    u_i_valid = u_i[valid]
    v_i_valid = v_i[valid]

    print(f"[CALIBRATION] Using {len(z_samples_valid)} SLI points (units now in mm)")

    # Print a small debug table of used points
    max_print = 20
    for idx in range(min(max_print, len(z_samples_valid))):
        print(
            f"  #{idx:3d}: "
            f"u={u_valid[idx]:7.2f} (idx {u_i_valid[idx]:3d}), "
            f"v={v_valid[idx]:7.2f} (idx {v_i_valid[idx]:3d}) | "
            f"z_rel={z_samples_valid[idx]:9.5f} -> Z_mm={Z_abs_valid[idx]:9.5f}"
        )
    if len(z_samples_valid) > max_print:
        print(f"  ... ({len(z_samples_valid) - max_print} more points)")

    # ---- Fit linear mapping Z_mm = a * z_rel + b ----
    if use_least_squares:
        A = np.vstack([z_samples_valid, np.ones_like(z_samples_valid)]).T
        (a, b), *_ = np.linalg.lstsq(A, Z_abs_valid, rcond=None)
    else:
        # Use just two points (min and max Z_mm)
        i_min = int(np.argmin(Z_abs_valid))
        i_max = int(np.argmax(Z_abs_valid))
        z1, Z1 = z_samples_valid[i_min], Z_abs_valid[i_min]
        z2, Z2 = z_samples_valid[i_max], Z_abs_valid[i_max]

        if z2 == z1:
            raise RuntimeError("Chosen calibration points have identical z_rel.")

        a = (Z2 - Z1) / (z2 - z1)
        b = Z1 - a * z1

    # ---- Apply scale/offset to full map (output is mm) ----
    z_abs_map = a * z_rel + b

    # ---- Save calibrated depth ----
    if output_path is None:
        output_path = depth_npy_path.with_name(depth_npy_path.stem + "_cal.npy")
    else:
        output_path = Path(output_path)

    np.save(output_path, z_abs_map.astype(np.float32))

    print(
        f"[CALIBRATION] depth: {depth_npy_path.name} -> {output_path.name} "
        f"in mm (a={a:.6g}, b={b:.6g})"
    )

    return z_abs_map, float(a), float(b)

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

