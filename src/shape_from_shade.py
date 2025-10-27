import cv2, numpy as np
from numpy.linalg import pinv

# --- Inputs you provide ---
# imgs: list of N grayscale float images (ambient-subtracted, same exposure), shape HxW each
# K: 3x3 camera intrinsics
# led_positions_cam: Nx3 array of LED centers in camera coordinates (meters)
# led_gains: length-N array (per-LED radiometric gains), e.g. ones if unknown
# z0: HxW coarse depth prior in meters (from your structure step)

def backproject_unit_rays(K, H, W):
    """Unit ray for each pixel in camera coords."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return r  # HxWx3

def compute_light_dirs_and_weights(z, rays, led_pos):
    """
    For each pixel and LED: unit direction to light and 1/r^2 falloff.
    Returns Lhat: HxWxNx3, falloff: HxWxN
    """
    H, W, _ = rays.shape
    N = led_pos.shape[0]
    # 3D point for each pixel
    X = z[...,None] * rays  # HxWx3
    # broadcast against LEDs
    r_i = led_pos[None,None,:,:] - X[...,None,:]      # HxWxNx3
    dist = np.linalg.norm(r_i, axis=-1)               # HxWxN
    Lhat = r_i / np.maximum(dist[...,None], 1e-6)     # HxWxNx3
    falloff = 1.0 / np.maximum(dist**2, 1e-9)         # HxWxN
    return Lhat, falloff

def nearfield_photometric_stereo(imgs, K, led_positions_cam, led_gains, z0):
    imgs = [i.astype(np.float32) for i in imgs]
    H, W = imgs[0].shape
    N = len(imgs)
    I = np.stack(imgs, axis=-1)                       # HxWxN

    rays = backproject_unit_rays(K, H, W)             # HxWx3
    Lhat, falloff = compute_light_dirs_and_weights(z0, rays, led_positions_cam)  # HxWxNx3

    # Apply per-LED gains into a single weight term
    calib = falloff * led_gains[None,None,:]          # HxWxN

    # Build per-pixel S matrices (N x 3)
    S = (Lhat * calib[...,None]).astype(np.float32)   # HxWxNx3

    # Solve g = rho*n by least squares per pixel using a precomputed pseudo-inverse
    # Pinv per pixel: (3xN) but computing per pixel is heavy; do batched using einsum.
    # Normal equations: (S^T S) g = S^T I
    STS = np.einsum('...ni,...nj->...ij', S, S)       # HxWx3x3
    ST_I = np.einsum('...ni,...n->...i', S, I)        # HxWx3

    # Solve 3x3 systems per pixel
    g = np.zeros((H,W,3), np.float32)
    # add small Tikhonov for stability
    lam = 1e-4
    Id3 = np.eye(3, dtype=np.float32)
    STS_reg = STS + lam*Id3
    # invert 3x3 per pixel
    det = np.linalg.det(STS_reg)
    valid = det > 1e-10
    STS_inv = np.zeros_like(STS_reg)
    STS_inv[valid] = np.linalg.inv(STS_reg[valid])
    g[valid] = np.einsum('...ij,...j->...i', STS_inv[valid], ST_I[valid])

    rho = np.linalg.norm(g, axis=2)
    n = g / np.maximum(rho[...,None], 1e-6)

    # shadow/specular mask
    mask = (rho > np.percentile(rho, 5)) & (n[...,2] > 0.15)

    return n, rho, mask