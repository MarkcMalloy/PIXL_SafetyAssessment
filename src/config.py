from pathlib import Path
### This class centralizes all configurable parameters.
### When new parameters are added, they should ultimately end up here
class Config:
    DEFAULT_INPUT_GLOB = "PIXL_Images/TestData/*.png"
    DEFAULT_OUTPUT_DIR = "../Output"
    OUTPUT_DIR_ALBEDO = "../Output/Albedo"
    OUTPUT_DIR_COMPOSITES = "../Output/Composites"
    OUTPUT_DIR_DEPTH = "../Output/Depth"
    OUTPUT_DIR_MASKS = "../Output/Masks"
    OUTPUT_DIR_NORMALIZATION = "../Output/Normalizations"
    OUTPUT_DIR_SHADOWS = "../Output/Shadows"
    DEFAULT_MORPH_OPEN_KSIZE = 3
    DEFAULT_MASK_QUANTILE = 0.55
    DEFAULT_PIXEL_SIZE = 1.0
    NUM_IMAGES = 6  # Expected number of input images to create a depth image
    LIGHT_ANGLES = [0, 60, 120, 180, 240, 300]  # Degrees for light directions
    Z_TILT = 1.5  # Z-component for light directions

    @staticmethod
    def ensure_dir(path: str):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)