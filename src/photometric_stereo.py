import numpy as np
from .config import Config

### Core logic for computing a photometric stereo image.
### TODO: Acquire the exact LED positions in mm and the exact position and angle/tilt of the camera to get much more accurate photometric computations
def build_light_dirs(angles_deg: list = Config.LIGHT_ANGLES, z_tilt: float = Config.Z_TILT) -> np.ndarray:
    """Build light directions for a ring around the camera."""
    angles = np.deg2rad(angles_deg)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    z = np.ones((len(angles), 1)) * z_tilt
    L = np.concatenate([xy, z], axis=1).astype(np.float32)
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-12
    return L

def solve_photometric_stereo(I: np.ndarray, L: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve for albedo and normals using photometric stereo."""
    H, W, K = I.shape
    if K != L.shape[0] or K != Config.NUM_IMAGES:
        # If we aren't using 6 images for Depth from Shade, then we wont compute a photometric stereo image
        raise ValueError(f"Expected {Config.NUM_IMAGES} images and lights, got {K} images and {L.shape[0]} lights")
    LT = L.T
    pinv = np.linalg.inv(LT @ L) @ LT
    I_reshaped = I.reshape(-1, K).T
    g = (pinv @ I_reshaped).T.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=-1)
    n = np.zeros_like(g, dtype=np.float32)
    nz = albedo > 1e-8
    n[nz] = (g[nz] / albedo[nz, None]).astype(np.float32)
    m = mask.astype(bool)
    albedo[~m] = 0.0
    n[~m] = 0.0
    flip = n[..., 2] < 0
    n[flip] = -n[flip]
    return albedo.astype(np.float32), n.astype(np.float32)