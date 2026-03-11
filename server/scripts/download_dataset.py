#!/usr/bin/env python3
"""Download a small USGS 3DEP LAZ tile for testing."""

from __future__ import annotations

import os
import urllib.request

# Small USGS 3DEP tile (~23MB LAZ) - Cameron Peak Wildfire, CO
URL = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/CO_CameronPeakWildfire_2021_D21/CO_CameronPkFire_1_2021/LAZ/USGS_LPC_CO_CameronPeakWildfire_2021_D21_w2945n1475.laz"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEST_FILE = os.path.join(DEST_DIR, "tile.laz")


def main() -> None:
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST_FILE):
        print(f"Already downloaded: {DEST_FILE}")
        return
    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, DEST_FILE)
    size_mb = os.path.getsize(DEST_FILE) / 1024 / 1024
    print(f"Downloaded {size_mb:.1f} MB to {DEST_FILE}")


if __name__ == "__main__":
    main()
