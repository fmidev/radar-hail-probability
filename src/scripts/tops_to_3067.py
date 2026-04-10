#!/usr/bin/env python
"""Reproject IRIS TOPS data from FIN1000 to EPSG:3067 (ETRS-TM35FIN) COG.

Output is uint8 with scale=100 (metres), offset=-100.
  raw=0:   undetect (no radar echo)
  raw=255: nodata   (outside coverage)
  raw=1–254: height = raw * 100 - 100  [0 – 25200 m]

Usage:
    python src/scripts/tops_to_3067.py legacy/data/CORECOMP.TOP /tmp/tops_3067.tif
    python src/scripts/tops_to_3067.py legacy/data/*.TOP /tmp/  # multiple files
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

from hailathon.io.iris import read_tops
from hailathon.projection.grid import CRS as FIN1000_CRS

DST_CRS = "EPSG:3067"
SCALE = 100.0
OFFSET = -100.0
NODATA = 255
UNDETECT = 0


def _src_bounds(da):
    x = da.coords["x"].values
    y = da.coords["y"].values
    dx = float(np.diff(x[:2])[0])
    dy = float(np.diff(y[:2])[0])
    return (x[0] - dx / 2, y[0] - dy / 2, x[-1] + dx / 2, y[-1] + dy / 2)


def tops_to_cog(src_path: str, dst_path: str) -> None:
    """Read an IRIS TOPS file and write a reprojected uint8 COG."""
    da = read_tops(src_path)
    height, width = da.shape
    left, bottom, right, top = _src_bounds(da)
    src_transform = from_bounds(left, bottom, right, top, width, height)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        FIN1000_CRS, DST_CRS, width, height,
        left=left, bottom=bottom, right=right, top=top,
    )

    # Reproject heights (float32, NaN for all masked)
    src_heights = da.values[::-1].astype(np.float32)
    dst_heights = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_heights, destination=dst_heights,
        src_transform=src_transform, src_crs=FIN1000_CRS,
        dst_transform=dst_transform, dst_crs=DST_CRS,
        resampling=Resampling.nearest,
        src_nodata=np.nan, dst_nodata=np.nan,
    )

    # Reproject noecho mask (uint8 0/1, nodata=255)
    src_noecho = da.coords["noecho"].values[::-1].astype(np.uint8)
    dst_noecho = np.full((dst_height, dst_width), 255, dtype=np.uint8)
    reproject(
        source=src_noecho, destination=dst_noecho,
        src_transform=src_transform, src_crs=FIN1000_CRS,
        dst_transform=dst_transform, dst_crs=DST_CRS,
        resampling=Resampling.nearest,
        src_nodata=255, dst_nodata=255,
    )

    # Encode to uint8
    raw = np.full((dst_height, dst_width), NODATA, dtype=np.uint8)
    valid = np.isfinite(dst_heights)
    scaled = np.round((dst_heights[valid] - OFFSET) / SCALE).astype(np.int32)
    raw[valid] = np.clip(scaled, 1, 254).astype(np.uint8)
    raw[dst_noecho == 1] = UNDETECT

    with rasterio.open(
        dst_path, "w",
        driver="COG",
        height=dst_height, width=dst_width, count=1,
        dtype="uint8", crs=DST_CRS,
        transform=dst_transform, nodata=NODATA,
    ) as dst:
        dst.write(raw, 1)
        dst.scales = [SCALE]
        dst.offsets = [OFFSET]
        dst.update_tags(source_file=Path(src_path).name, units="m")
        dst.update_tags(1, undetect=str(UNDETECT))

    print(f"  {Path(src_path).name} -> {dst_path}  ({Path(dst_path).stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Reproject TOPS to EPSG:3067 COG")
    parser.add_argument("inputs", nargs="+", help="IRIS TOPS file(s)")
    parser.add_argument("output", help="Output .tif path or directory for multiple inputs")
    args = parser.parse_args()

    out = Path(args.output)
    if len(args.inputs) > 1 or out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for src in args.inputs:
            dst = out / (Path(src).stem + "_3067.tif")
            tops_to_cog(src, str(dst))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        tops_to_cog(args.inputs[0], str(out))


if __name__ == "__main__":
    main()
