#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import Config

DEBUG_PRINT = True


def load_calibrated_depth() -> np.ndarray:
    depth_path = Path(Config.OUTPUT_DIR_DEPTH) / "cal_depth.npy"
    if not depth_path.is_file():
        raise FileNotFoundError(f"calibrated depth not found: {depth_path}")

    z = np.load(depth_path).astype(np.float64)
    z = np.fliplr(z)

    return z


def load_sli_csv() -> tuple[pd.DataFrame, Path]:
    csv_path = Config.resolve_single_csv(Config.DEFAULT_SLI_CSV_GLOB_40mm)
    if csv_path is None:
        raise FileNotFoundError("No SLI CSV found via Config glob")

    df = pd.read_csv(csv_path)

    required = {"u_px", "v_px", "Z_um"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in SLI CSV: {missing}")

    df["Z_mm"] = df["Z_um"] / 1000.0
    return df, csv_path


def compute_depth_rms(z_map: np.ndarray, df: pd.DataFrame):
    H, W = z_map.shape

    errors = []
    debug_rows = []

    for i, row in df.iterrows():
        u = int(round(row["u_px"]))
        v = int(round(row["v_px"]))
        z_ref = float(row["Z_mm"])

        v = (H - 1) - v

        if not (0 <= u < W and 0 <= v < H):
            continue

        z_cal = z_map[v, u]
        if not np.isfinite(z_cal):
            continue

        err = z_cal - z_ref
        errors.append(err)

        if DEBUG_PRINT:
            debug_rows.append((i, u, v, z_cal, z_ref, err))

    if not errors:
        raise RuntimeError("No valid SLI points for RMS computation")

    errors = np.asarray(errors, dtype=np.float64)

    rms = float(np.sqrt(np.mean(errors ** 2)))
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    max_abs = float(np.max(np.abs(errors)))

    if DEBUG_PRINT:
        print("\n--- RMS DEBUG ---")
        print("Idx   u    v     Z_cal(mm)   Z_ref(mm)    err(mm)    err^2")
        print("-" * 72)
        for i, u, v, zc, zr, e in debug_rows:
            print(
                f"{i:3d} {u:4d} {v:4d} "
                f"{zc:11.4f} {zr:11.4f} {e:10.4f} {e*e:10.6f}"
            )

        print("\n--- AGGREGATES ---")
        print(f"N                 = {len(errors)}")
        print(f"Sum(err)          = {errors.sum():.6f}")
        print(f"Mean(err)         = {mean:.6f}")
        print(f"Sum(err^2)        = {(errors**2).sum():.6f}")
        print(f"Mean(err^2)       = {(errors**2).mean():.6f}")
        print(f"RMS               = {rms:.6f}")
        print(f"Std(err)          = {std:.6f}")
        print(f"Max |err|         = {max_abs:.6f}")

    return rms, mean, std, max_abs, len(errors)


def main():
    print("\n=== Depth RMS vs SLI ===")

    depth_path = Path(Config.OUTPUT_DIR_DEPTH) / "cal_depth.npy"
    df, csv_path = load_sli_csv()

    print(f"Depth file: {depth_path}")
    print(f"SLI CSV   : {csv_path}")

    z_map = load_calibrated_depth()

    rms, mean, std, max_abs, n = compute_depth_rms(z_map, df)

    print("\n=== SUMMARY ===")
    print(f"Points used : {n}")
    print(f"RMS error   : {rms:.4f} mm")
    print(f"Mean error  : {mean:.4f} mm")
    print(f"Std error   : {std:.4f} mm")
    print(f"Max |err|   : {max_abs:.4f} mm")


if __name__ == "__main__":
    main()
