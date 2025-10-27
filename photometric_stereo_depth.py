from pathlib import Path
import os
import numpy as np
import cv2


def ensure_dir(p:str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_pngs(folder_or_glob:str):
    """Load 6 grayscale PNGs sorted by name. Returns HxWx6 float32 in [0,1]."""
    p = Path(folder_or_glob)
    if p.is_dir():
        files = sorted([str(f) for f in p.glob("*.png")])
    else:
        files = sorted([str(f) for f in Path().glob(folder_or_glob)])
    if len(files) < 6:
        raise ValueError(f"Expected at least 6 PNGs, found {len(files)} at {folder_or_glob}")
    files = files[:6]
    imgs = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Failed to read: {f}")
        imgs.append(im.astype(np.float32) / 255.0)
    I = np.stack(imgs, axis=-1)  # (H,W,6)
    return I, files


def normalize_uint8(img:np.ndarray):
    """Normalize an image to [0,255] uint8 for visualization (ignores NaNs)."""
    m = np.isfinite(img)
    if not np.any(m):
        return np.zeros_like(img, dtype=np.uint8)
    a = img[m].min()
    b = img[m].max()
    if b <= a + 1e-12:
        out = np.zeros_like(img, dtype=np.uint8)
        out[m] = 0
        return out
    out = np.zeros_like(img, dtype=np.float32)
    out[m] = (img[m]-a)/(b-a)
    out = np.clip(out*255.0,0,255).astype(np.uint8)
    return out


def otsu_on_max(I:np.ndarray, morph_open_ksize:int=3):
    """Foreground mask from Otsu thresholding on max-intensity composite."""
    Imax = I.max(axis=-1)
    I8 = normalize_uint8(Imax)
    # Otsu
    thr, mask = cv2.threshold(I8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (mask>0).astype(np.uint8)
    # Morph open to remove speckles
    if morph_open_ksize>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask, Imax


def build_default_light_dirs():
    """
    6 light directions approximating a ring around the camera, pointing mostly along +Z.
    Adjust to your real geometry for best results.
    """
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])  # around Z
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    z = np.ones((6,1))*1.5  # slightly tilted towards +Z; increase to make lights more frontal
    L = np.concatenate([xy, z], axis=1).astype(np.float32)
    # normalize rows
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-12
    return L  # (6,3)


def solve_photometric_stereo(I:np.ndarray, L:np.ndarray, mask:np.ndarray):
    """
    I: (H,W,6) intensities in [0,1]
    L: (6,3) light directions (unit)
    mask: (H,W) 0/1
    returns:
      albedo (H,W), normals (H,W,3) with unit length where mask==1 else 0
    """
    H, W, K = I.shape
    assert K == L.shape[0] == 6, "This implementation assumes 6 images/lights"
    # Precompute pseudoinverse
    LT = L.T
    pinv = np.linalg.inv(LT @ L) @ LT  # (3,6)
    I_reshaped = I.reshape(-1, K).T  # (6, H*W)
    g = (pinv @ I_reshaped).T  # (H*W, 3)
    g = g.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=-1)
    n = np.zeros_like(g, dtype=np.float32)
    nz = albedo > 1e-8
    n[nz] = (g[nz] / albedo[nz, None]).astype(np.float32)
    # enforce mask
    m = mask.astype(bool)
    albedo[~m] = 0.0
    n[~m] = 0.0
    # flip normals to face camera (+Z)
    flip = n[...,2] < 0
    n[flip] = -n[flip]
    return albedo.astype(np.float32), n.astype(np.float32)


def frankot_chellappa(p:np.ndarray, q:np.ndarray):
    """
    Integrate gradients p = dz/dx, q = dz/dy into a surface z using Frankot–Chellappa.
    Returns z with zero-mean.
    """
    H, W = p.shape
    # Frequencies
    wx = np.fft.fftfreq(W) * 2*np.pi
    wy = np.fft.fftfreq(H) * 2*np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + WY**2
    denom[0,0] = 1.0  # avoid div by zero
    # Fourier transforms
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    Z = (-1j*WX*P - 1j*WY*Q) / denom
    Z[0,0] = 0.0  # set DC to 0
    z = np.real(np.fft.ifft2(Z)).astype(np.float32)
    z -= np.nanmean(z)
    return z


def normals_to_depth(n:np.ndarray, mask:np.ndarray, pixel_size:float=1.0):
    """
    Convert normals to p=dz/dx, q=dz/dy, then integrate.
    n: (H,W,3), unit
    mask: (H,W) 0/1
    pixel_size: scale factor for gradients if pixels have physical size.
    """
    eps = 1e-8
    nz = n[...,2]
    nz = np.where(nz==0, eps, nz)
    p = -n[...,0] / (nz+eps)  # dz/dx
    q = -n[...,1] / (nz+eps)  # dz/dy
    p *= pixel_size
    q *= pixel_size
    # Zero out outside mask to stabilize
    m = mask.astype(bool)
    p[~m] = 0.0
    q[~m] = 0.0
    z = frankot_chellappa(p, q)
    # Set outside mask to NaN for clarity
    z = z.astype(np.float32)
    z[~m] = np.nan
    return z


def save_normals_rgb(n:np.ndarray, path:str):
    """Map normals from [-1,1] to [0,255] for RGB visualization."""
    nn = (n + 1.0) * 0.5
    img = np.clip(nn * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_shadow_maps(n:np.ndarray, L:np.ndarray, mask:np.ndarray, out_dir:str):
    """
    Per-light shadow = (n · s_i <= 0) OR very dark pixel heuristic.
    Writes 6 PNG masks to out_dir/shadow_i.png
    """
    ensure_dir(out_dir)
    H, W, _ = n.shape
    for i in range(L.shape[0]):
        s = L[i]  # (3,)
        ndotl = (n[...,0]*s[0] + n[...,1]*s[1] + n[...,2]*s[2])
        shadow = (ndotl <= 0).astype(np.uint8) * 255
        # Also remove background
        shadow[mask==0] = 0
        cv2.imwrite(str(Path(out_dir)/f"shadow_{i}.png"), shadow)


def main(
    input_glob_or_folder="PIXL_Images/TestData/*.png",
    output_dir="Output",
    use_otsu=True,
    mask_quantile=0.55  # used if use_otsu=False
):
    ensure_dir(output_dir)
    I, files = load_pngs(input_glob_or_folder)
    H, W, _ = I.shape

    # Mask
    if use_otsu:
        mask, Imax = otsu_on_max(I, morph_open_ksize=3)
    else:
        Imax = I.max(axis=-1)
        t = np.quantile(Imax, mask_quantile)
        mask = (Imax > t).astype(np.uint8)

    # Light directions (adjust to real geometry!)
    L = build_default_light_dirs()  # (6,3)

    # Solve albedo and normals
    albedo, n = solve_photometric_stereo(I, L, mask)

    # Depth from normals
    z = normals_to_depth(n, mask, pixel_size=1.0)

    # Write outputs
    ensure_dir(output_dir)
    cv2.imwrite(str(Path(output_dir)/"albedo.png"), normalize_uint8(albedo))
    save_normals_rgb(n, str(Path(output_dir)/"normals.png"))
    # Depth (two variants: normalized PNG for viz, and float32 EXR for data)
    cv2.imwrite(str(Path(output_dir)/"depth.png"), normalize_uint8(np.nan_to_num(z, nan=0.0)))
    #cv2.imwrite(str(Path(output_dir)/"depth.exr"), np.nan_to_num(z, nan=0.0).astype(np.float32))
    cv2.imwrite(str(Path(output_dir) / "depth.pfm"), np.nan_to_num(z, nan=0.0).astype(np.float32))

    # Shadow maps
    save_shadow_maps(n, L, mask, str(Path(output_dir)/"Shadows"))

    # Also export the mask and a composite
    cv2.imwrite(str(Path(output_dir)/"mask.png"), (mask*255).astype(np.uint8))
    cv2.imwrite(str(Path(output_dir)/"composite_max.png"), normalize_uint8(I.max(axis=-1)))

    print("Processed files (first 6):")
    for f in files[:6]:
        print(" -", f)
    print("Wrote outputs to:", Path(output_dir).resolve())


if __name__ == "__main__":
    main()
