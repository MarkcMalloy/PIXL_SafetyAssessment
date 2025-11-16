from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


# ---------- header struct ----------

@dataclass
class Header:
    DAC_OFFSET: int
    DAC_gain: int
    INTEGRATION_TIME: int
    COMPRESSION: int
    ROI: int
    JPEG_QUALITY: int
    COMPRESSION_THRESHOLD: int
    INFO: int
    VALID: int
    STATUS: int
    CODE_START: int
    CODE_END: int
    SUB_TIMESTAMP: int
    TIMESTAMP: int
    H: int
    W: int
    IMOD: int


# ---------- low-level readers ----------

def _read_uint16(f, n: int) -> np.ndarray:
    # little-endian uint16
    return np.fromfile(f, dtype="<u2", count=n)


def _read_uint8(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype="u1", count=n)


def _read_uint32(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype="<u4", count=n)


def read_header(f) -> Header:
    temp = _read_uint16(f, 2)
    if temp.size < 2:
        raise EOFError("File too short to contain header")

    # hexapod metadata compensation
    if temp[0] == 65535:
        _ = _read_uint16(f, 14)  # skip 7*2 uint16
        temp = _read_uint16(f, 2)

    DAC_OFFSET, DAC_gain = map(int, temp)

    temp8 = _read_uint8(f, 6)
    INTEGRATION_TIME = int(temp8[0])
    COMPRESSION = int(temp8[1])
    ROI = int(temp8[2])
    JPEG_QUALITY = int(temp8[3])
    COMPRESSION_THRESHOLD = int(temp8[4])
    INFO = int(temp8[5])

    temp16 = _read_uint16(f, 2)
    VALID, STATUS = map(int, temp16)

    temp32 = _read_uint32(f, 2)
    CODE_START, CODE_END = map(int, temp32)

    SUB_TIMESTAMP = int(_read_uint16(f, 1)[0])
    TIMESTAMP = int(_read_uint32(f, 1)[0])

    temp16 = _read_uint16(f, 3)
    H, W, IMOD = map(int, temp16)

    return Header(
        DAC_OFFSET=DAC_OFFSET,
        DAC_gain=DAC_gain,
        INTEGRATION_TIME=INTEGRATION_TIME,
        COMPRESSION=COMPRESSION,
        ROI=ROI,
        JPEG_QUALITY=JPEG_QUALITY,
        COMPRESSION_THRESHOLD=COMPRESSION_THRESHOLD,
        INFO=INFO,
        VALID=VALID,
        STATUS=STATUS,
        CODE_START=CODE_START,
        CODE_END=CODE_END,
        SUB_TIMESTAMP=SUB_TIMESTAMP,
        TIMESTAMP=TIMESTAMP,
        H=H,
        W=W,
        IMOD=IMOD,
    )


# ---------- mat2gray equivalent ----------

def mat2gray_np(im: np.ndarray) -> np.ndarray:
    """
    MATLAB-like mat2gray:
    - convert to float
    - scale to [0,1] using global min/max
    """
    im = im.astype(np.float32)
    imin = im.min()
    imax = im.max()
    if imax > imin:
        im_norm = (im - imin) / (imax - imin)
    else:
        im_norm = np.zeros_like(im, dtype=np.float32)
    return im_norm


# ---------- per-file processing ----------

def load_unc_full_image(path: Path) -> np.ndarray | None:
    """
    Reads a .unc/.unk file and returns the full `im` array
    (H*(IMOD+1) x W) for COMPRESSION == 0, or None otherwise.
    Mirrors the 'unc' case in readAscImage.m.
    """
    with open(path, "rb") as f:
        header = read_header(f)

        if header.COMPRESSION != 0:
            # not an 'unc' image
            return None

        n = header.H * (header.IMOD + 1) * header.W
        temp = _read_uint8(f, n)
        if temp.size < 4:
            return None

        # TRN case: first bytes are "$TF$" → not a plain image
        if temp[0] == 36 and temp[1] == 84 and temp[2] == 70 and temp[3] == 36:
            return None

        # MATLAB: im = reshape(temp, W, H*(IMOD+1))';
        im = temp.reshape((header.W, header.H * (header.IMOD + 1))).T
        return im


def process_directory(input_dir: str | Path, output_dir: str | Path):
    """
    For every .unc file in input_dir:
      - read full im array
      - apply mat2gray
      - save as <stem>_full.png in output_dir
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.unc"))
    if not files:
        print(f"No .unc files found in {input_dir}")
        return

    for path in files:
        im = load_unc_full_image(path)
        if im is None:
            print(f"[INFO] {path.name}: not a plain 'unc' image, skipped")
            continue

        # mat2gray + scale to 0–255 + uint8
        im_norm = mat2gray_np(im)
        I8 = (im_norm * 255.0 + 0.5).astype(np.uint8)

        out_name = f"{path.stem}_full.png"
        out_path = output_dir / out_name
        Image.fromarray(I8, mode="L").save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    in_dir = r"PIXL_Images\CalData\PIXL_040mm_dist\NoObstacle"
    out_dir = r"PIXL_Images/CalData/Output/40mm_NoObstacle_full"
    process_directory(in_dir, out_dir)
