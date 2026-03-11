from __future__ import annotations

import argparse
import asyncio
import logging

from .server import start_server
from .terrain import generate_terrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxel WebSocket Server")
    parser.add_argument("--file", help="LAS/LAZ file to load")
    parser.add_argument("--resolution", type=float, default=1.0, help="World units per voxel")
    parser.add_argument("--port", type=int, default=9876, help="WebSocket port")
    parser.add_argument(
        "--terrain-size",
        type=int,
        default=256,
        help="Procedural terrain size (if no file given)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.file:
        from .ingest import ingest_las

        print(f"Loading {args.file} at resolution {args.resolution}...")
        grid = ingest_las(args.file, args.resolution)
    else:
        print(f"Generating procedural terrain ({args.terrain_size}x{args.terrain_size})...")
        grid = generate_terrain(args.terrain_size)

    print(f"Loaded {len(grid.patch_lods)} patches, serving on port {args.port}")
    asyncio.run(start_server(grid, "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
