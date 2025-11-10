import numpy as np
from pathlib import Path
from .image_io import save_image
from .config import Config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

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

def save_depth_plot(z: np.ndarray, mask: np.ndarray, out_path: str):
    """
    Create a 3D surface plot from a depth map and save it as an image.

    Parameters:
        z       : np.ndarray, shape (H, W)
                  Depth map.
        mask    : np.ndarray, shape (H, W)
                  Boolean mask of valid pixels.
        out_path: str
                  Path to save the output image.
    """
    # Mask out invalid areas
    z_masked = np.where(mask, z, np.nan)
    H, W = z_masked.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(X, Y, z_masked, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Labels and view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.view_init(elev=30, azim=120)

    # Save figure
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)