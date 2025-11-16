from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ----------------- Data containers ----------------- #

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


@dataclass
class SLIHeader:
    Ncen: int
    Nmatch: int
    RMSE: float
    planeN: np.ndarray  # (3,)
    planeD: int
    planeRMSE: int
    warning: int
    error: int


@dataclass
class SLIData:
    cen: np.ndarray      # (N, 3), raw fixed-point coords + extra
    Q: np.ndarray        # (N, 3), int32
    sliID: np.ndarray    # (N,)
    residual: np.ndarray # (N,)


@dataclass
class AscImage:
    header: Header
    image: np.ndarray | None   # camera image for 'unc' (H*(IMOD+1), W) or None
    sliHeader: SLIHeader | None
    sliData: SLIData | None


# ----------------- Low-level readers ----------------- #

def _read_uint16(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype=np.uint16, count=n)


def _read_uint8(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype=np.uint8, count=n)


def _read_uint32(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype=np.uint32, count=n)


def _read_int32(f, n: int) -> np.ndarray:
    return np.fromfile(f, dtype=np.int32, count=n)


def read_header(f) -> Header:
    """
    Mirror the header-reading part of readAscImage.m
    """
    temp = _read_uint16(f, 2)
    if temp.size < 2:
        raise EOFError("File too short to contain header.")

    # Hexapod metadata compensation
    if temp[0] == 65535:
        _ = _read_uint16(f, 2 * 7)  # skip 14 uint16
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


def read_unc_image(f, header: Header) -> np.ndarray | None:
    """
    COMPRESSION == 0 case ('unc'):
    Reads raw uint8 image unless it's a $TF$ TRN file (then we
    simply return None for now, as your CSVs are based on SLI).
    """
    n = header.H * (header.IMOD + 1) * header.W
    temp = _read_uint8(f, n)
    if temp.size < 4:
        return None

    # Check for "$TF$" at the start
    if temp[0] == 36 and temp[1] == 84 and temp[2] == 70 and temp[3] == 36:
        # TRN data – not needed for SLI CSVs, so we skip
        return None

    # MATLAB: reshape(temp, W, H*(IMOD+1))'
    img = temp.reshape((header.W, header.H * (header.IMOD + 1))).T
    return img


def read_sli_block(f, header: Header) -> tuple[SLIHeader, SLIData]:
    """
    COMPRESSION == 34 case ('sli').

    In your dataset, cen(:,0:1) are stored as fixed-point pixel
    coordinates (Q12-style): true_pixel = raw / 4096.
    We store the raw values here and convert later in export_sli_csv.
    """

    Ncen = int(_read_uint8(f, 1)[0])
    Nmatch = int(_read_uint8(f, 1)[0])
    RMSE = float(_read_uint8(f, 1)[0]) / 10.0
    _ = _read_uint8(f, 1)  # unused temp

    planeN = _read_int32(f, 3).astype(np.float64) / 2147483647.0
    planeD = int(_read_uint32(f, 1)[0])
    planeRMSE = int(_read_uint16(f, 1)[0])
    _ = _read_uint16(f, 1)  # unused temp
    warning = int(_read_uint8(f, 1)[0])
    error = int(_read_uint8(f, 1)[0])

    sli_header = SLIHeader(
        Ncen=Ncen,
        Nmatch=Nmatch,
        RMSE=RMSE,
        planeN=planeN,
        planeD=planeD,
        planeRMSE=planeRMSE,
        warning=warning,
        error=error,
    )

    # Now read per-centre data
    cen = np.zeros((Ncen, 3), dtype=np.float64)
    Q = np.zeros((Ncen, 3), dtype=np.int32)
    sliID = np.zeros(Ncen, dtype=np.uint8)
    residual = np.zeros(Ncen, dtype=np.float64)

    for i in range(Ncen):
        # Raw fixed-point pixel coords (Q12-like)
        c1_raw = _read_uint32(f, 1)[0]
        c2_raw = _read_uint32(f, 1)[0]
        c3 = _read_uint16(f, 1)[0]
        cen[i, :] = (float(c1_raw), float(c2_raw), float(c3))

        # Q(i,1:3) = int32
        Q[i, :] = _read_int32(f, 3)

        # sliID(i,1) = uint8
        sliID[i] = _read_uint8(f, 1)[0]

        # residual(i,1) = uint8 / 10
        residual[i] = _read_uint8(f, 1)[0] / 10.0

    sli_data = SLIData(cen=cen, Q=Q, sliID=sliID, residual=residual)
    return sli_header, sli_data


def read_asc_image(path: str | Path) -> AscImage:
    """
    Python port of the subset of readAscImage.m you need:
    - full header
    - 'unc' and 'sli' image formats
    Other formats ('jp0', 'cen', 'roi', etc.) are not parsed and
    will have image/sliData = None.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = read_header(f)

        image = None
        sli_header = None
        sli_data = None

        if header.COMPRESSION == 0:      # 'unc'
            image = read_unc_image(f, header)

        elif header.COMPRESSION == 34:   # 'sli'
            sli_header, sli_data = read_sli_block(f, header)

        else:
            # For now we ignore other formats ('jp0', 'cen', 'roi', etc.)
            pass

    return AscImage(
        header=header,
        image=image,
        sliHeader=sli_header,
        sliData=sli_data,
    )


# ----------------- CSV export ----------------- #

def export_sli_csv(im: AscImage, csv_path: str | Path):
    """
    Export SLI points to CSV with columns:
        u,v,X,Y,Z

    im.sliData.cen(:,0:1) are stored as fixed-point pixel coordinates:
        u_pixel = cen[:,0] / 4096
        v_pixel = cen[:,1] / 4096

    X,Y,Z come directly from Q.
    """
    if im.sliData is None or im.sliHeader is None:
        raise ValueError("AscImage has no SLI data to export.")

    header = im.header
    H, W = header.H, header.W

    # Optional sanity check
    if (H, W) not in [(580, 752), (752, 580)]:
        print(f"Warning: header.H/W = ({H},{W}) not equal to 580x752.")

    cen = im.sliData.cen  # (N,3), raw fixed-point
    Q = im.sliData.Q      # (N,3)

    # Convert fixed-point to pixel coordinates (float)
    u = cen[:, 0] / 4096.0
    v = cen[:, 1] / 4096.0
    print("u: ",u/4096)
    print("v: ",v/4096)

    X = Q[:, 0].astype(np.float64)
    Y = Q[:, 1].astype(np.float64)
    Z = Q[:, 2].astype(np.float64)

    data = np.column_stack([u, v, X, Y, Z])

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header_line = "u,v,X,Y,Z"
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header_line,
        comments="",
        fmt="%.9f",
    )
    print(f"Saved SLI CSV: {csv_path}")


# ----------------- Batch processing ----------------- #

def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
):
    """
    Process all .unk and .unc files in input_dir.

    For every file whose header.COMPRESSION == 34 ('sli'), write:
        <output_dir>/<stem>_SLI_points.csv
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.unc")) + list(input_dir.glob("*.unk")))
    if not files:
        print(f"No .unc or .unk files found in {input_dir}")
        return

    for path in files:
        try:
            asc = read_asc_image(path)
        except Exception as e:
            print(f"[SKIP] {path.name}: error while reading ({e})")
            continue

        if asc.header.COMPRESSION == 34 and asc.sliData is not None:
            csv_name = f"{path.stem}_SLI_points.csv"
            csv_path = output_dir / csv_name
            export_sli_csv(asc, csv_path)
        else:
            # For now we just report non-SLI frames
            print(f"[INFO] {path.name}: COMPRESSION={asc.header.COMPRESSION} (no SLI export)")


if __name__ == "__main__":
    # Your paths
    in_dir = r"PIXL_Images\CalData\PIXL_040mm_dist\NoObstacle"
    out_dir = r"PIXL_Images/CalData/Output/40mm_NoObstacle"

    process_directory(in_dir, out_dir)
