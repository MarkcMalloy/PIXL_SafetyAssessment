from __future__ import annotations

from pathlib import Path

class Config:
    # --- Project root resolution ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --- Input / Output paths ---
    #OBJECTTYPE = "noObstacle"
    OBJECTTYPE = "obstacle"
    #OBJECTTYPE = "Calibration"

    #DEFAULT_INPUT_GLOB = str(PROJECT_ROOT / "PIXL_Images" / "TestData" / "*.png")
    DEFAULT_INPUT_GLOB_40mm = str(PROJECT_ROOT / "PIXL_Images" / "CalData" / "PIXL_040mm_dist" / OBJECTTYPE / "images" / "*.png")
    DEFAULT_SLI_CSV_GLOB_40mm = str(PROJECT_ROOT / "PIXL_Images" / "CalData" / "PIXL_040mm_dist" / OBJECTTYPE / "*.csv")

    DEFAULT_INPUT_GLOB_100mm = str(PROJECT_ROOT / "PIXL_Images" / "CalData" / "PIXL_100mm_dist" / OBJECTTYPE / "images" / "*.png")
    DEFAULT_SLI_CSV_GLOB_100mm = str(PROJECT_ROOT / "PIXL_Images" / "CalData" / "PIXL_100mm_dist" / OBJECTTYPE / "*.csv")



    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "Output")

    OUTPUT_DIR_ALBEDO = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE / "Albedo")
    OUTPUT_DIR_COMPOSITES = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE / "Composites")
    OUTPUT_DIR_DEPTH = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE / "Depth")
    OUTPUT_DIR_MASKS = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE / "Masks")
    OUTPUT_DIR_NORMALIZATION = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE / "Normalizations")
    OUTPUT_DIR_SHADOWS = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE /"Shadows")
    OUTPUT_DIR_SEGMENTATION = str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR / OBJECTTYPE /"Segmentation")


    # --- Photometric parameters ---
    DEFAULT_MORPH_OPEN_KSIZE = 3
    DEFAULT_MASK_QUANTILE = 0.55
    DEFAULT_PIXEL_SIZE = 1.0
    NUM_IMAGES = 6  # Expected number of input images to create a depth image
    LIGHT_ANGLES = [0, 60, 120, 180, 240, 300]  # Degrees for light directions
    Z_TILT = 2  # Z-component for light directions. It is the relative geometric offset between the LED light positions and the camera's optical axis
    # Each light vector before normalization can be described as v_i = [cos(theta_i),sin(theta_i),z_tilt]. Z-tilt can be described as height of the LED ring / radius of the ring
    # At Z_TILT = 2, the lights are almost coaxial with the camera

    @classmethod
    def output_path(cls, subdir: str) -> Path:
        path = cls.PROJECT_ROOT / "Output" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def ensure_dir(path: str):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve_single_csv(csv_glob: str) -> Path | None:
        """
        Accepts absolute or relative globs like:
          C:/.../PIXL_040mm_dist/obstacle/*.csv
          PIXL_Images/CalData/.../*.csv
        Returns the single CSV Path if exactly one exists.
        """
        p = Path(csv_glob)

        # If they passed a glob (like *.csv), p.name is the pattern and p.parent is the folder
        folder = p.parent
        pattern = p.name

        print(f"Finding csv in: {folder} pattern: {pattern}")

        if not folder.exists():
            return None

        matches = sorted(folder.glob(pattern))

        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise RuntimeError(
                "Expected exactly 1 CSV but found multiple:\n"
                + "\n".join(f" - {m}" for m in matches)
            )
        return matches[0]
