import os
import glob
import cv2
import numpy as np

def try_load(path_candidates):
    """Return first successfully loaded .npy from the candidates, else None."""
    for p in path_candidates:
        if os.path.isfile(p):
            try:
                return np.load(p)
            except Exception:
                pass
    return None

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


def save_shadow_maps(n, z0, K, led_positions_cam, out_dir):
    """
    Compute per-light shadow maps using n·l <= 0 with the current z0 geometry.
    Saves HxW 8-bit PNGs to out_dir (Shadow_00.png ...).
    """
    H, W, _ = n.shape
    # Recreate rays and light directions for current geometry
    rays = backproject_unit_rays(K, H, W)  # HxWx3
    Lhat, _ = compute_light_dirs_and_weights(z0, rays, led_positions_cam)  # HxWxNx3

    # dot(n, Lhat) -> HxWxN
    # (einsum over last axis of n and last axis of Lhat)
    ndotl = np.einsum('hwi,hwni->hwn', n, Lhat)

    shadows = (ndotl <= 0.0).astype(np.uint8) * 255  # 0 or 255 per pixel
    os.makedirs(out_dir, exist_ok=True)
    N = shadows.shape[-1]
    for i in range(N):
        out_path = os.path.join(out_dir, f"Shadow_{i:02d}.png")
        cv2.imwrite(out_path, shadows[..., i])

def main():
    # 1) Load six PNGs (grayscale float32)
    image_folder = os.path.join('PIXL_Images', 'SOL 1654')
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    if len(image_paths) < 6:
        raise ValueError(f"Expected at least 6 images, found {len(image_paths)} in {image_folder}")

    image_paths = image_paths[:6]
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    if any(im is None for im in imgs):
        bad = [p for p, im in zip(image_paths, imgs) if im is None]
        raise IOError(f"Failed to read some images: {bad}")
    imgs = [im.astype(np.float32) / 255.0 for im in imgs]
    H, W = imgs[0].shape
    N = len(imgs)  # should be 6

    # 2) Camera intrinsics K (try load, else reasonable placeholder)
    def try_load(path_candidates):
        for p in path_candidates:
            if os.path.isfile(p):
                try:
                    return np.load(p)
                except Exception:
                    pass
        return None

    K = try_load(['K.npy', os.path.join(image_folder, 'K.npy')])
    if K is None:
        f = float(max(H, W))
        K = np.array([[f, 0, (W - 1) / 2.0],
                      [0, f, (H - 1) / 2.0],
                      [0, 0, 1.0]], dtype=np.float32)
    else:
        K = K.astype(np.float32)

    # 3) LED positions: ring (radius 0.04 m), centered at camera, z=0.
    # Order matches images: start at bottom-left, then move clockwise.
    # Camera coords in this code: x->right, y->down, z->forward.
    r = 0.04  # meters (8 cm diameter => 4 cm radius)
    start_deg = 225.0  # bottom-left
    step_deg = -360.0 / N  # clockwise steps
    thetas = np.deg2rad([start_deg + k * step_deg for k in range(N)])
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    zs = np.zeros_like(xs)
    led_positions_cam = np.stack([xs, ys, zs], axis=1).astype(np.float32)  # Nx3

    # 4) LED gains (try load, else ones)
    led_gains = try_load(['led_gains.npy', os.path.join(image_folder, 'led_gains.npy')])
    if led_gains is None:
        led_gains = np.ones((N,), dtype=np.float32)
    else:
        led_gains = led_gains.astype(np.float32)
    if led_gains.shape[0] != N:
        raise ValueError(f"Expected {N} LED gains, got {led_gains.shape[0]}")

    # 5) Depth prior z0 (try load, else flat plane at 0.10 m)
    z0 = try_load(['z0.npy', os.path.join(image_folder, 'z0.npy')])
    if z0 is None:
        z0 = np.full((H, W), 0.10, dtype=np.float32)  # 10 cm from camera
    else:
        z0 = z0.astype(np.float32)
    if z0.shape != (H, W):
        raise ValueError(f"z0 shape mismatch: expected {(H, W)}, got {z0.shape}")

    # 6) Run near-field photometric stereo
    n, rho, mask = nearfield_photometric_stereo(imgs, K, led_positions_cam, led_gains, z0)

    # 7) Save per-light shadow maps (n·l <= 0) to Output/Shadows
    out_dir = os.path.join('Output', 'Shadows')
    save_shadow_maps(n, z0, K, led_positions_cam, out_dir)

    # (Optional) Save results
    # os.makedirs('Output', exist_ok=True)
    # np.save(os.path.join('Output', 'normals.npy'), n)
    # np.save(os.path.join('Output', 'albedo.npy'), rho)
    # np.save(os.path.join('Output', 'mask.npy'), mask)

if __name__ == "__main__":
    main()