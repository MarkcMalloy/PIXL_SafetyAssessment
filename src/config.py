from pathlib import Path

class Config:
    # --- Project root resolution ---
    # This automatically finds the top-level folder (one up from /src)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --- Input / Output paths ---
    DEFAULT_INPUT_GLOB = str(PROJECT_ROOT / "PIXL_Images" / "TestData" / "*.png")
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "Output")

    OUTPUT_DIR_ALBEDO = str(PROJECT_ROOT / "Output" / "Albedo")
    OUTPUT_DIR_COMPOSITES = str(PROJECT_ROOT / "Output" / "Composites")
    OUTPUT_DIR_DEPTH = str(PROJECT_ROOT / "Output" / "Depth")
    OUTPUT_DIR_MASKS = str(PROJECT_ROOT / "Output" / "Masks")
    OUTPUT_DIR_NORMALIZATION = str(PROJECT_ROOT / "Output" / "Normalizations")
    OUTPUT_DIR_SHADOWS = str(PROJECT_ROOT / "Output" / "Shadows")

    # --- Photometric parameters ---
    DEFAULT_MORPH_OPEN_KSIZE = 3
    DEFAULT_MASK_QUANTILE = 0.55
    DEFAULT_PIXEL_SIZE = 1.0
    NUM_IMAGES = 6  # Expected number of input images to create a depth image
    LIGHT_ANGLES = [0, 60, 120, 180, 240, 300]  # Degrees for light directions
    Z_TILT = 1.5  # Z-component for light directions

    @classmethod
    def output_path(cls, subdir: str) -> Path:
        path = cls.PROJECT_ROOT / "Output" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def ensure_dir(path: str):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)
