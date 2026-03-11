from __future__ import annotations

import argparse
import asyncio
import logging
import time

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
        default=4096,
        help="Procedural terrain size (if no file given)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.INFO)

    t0 = time.perf_counter()
    if args.file:
        from .ingest import ingest_las

        print(f"Loading {args.file} at resolution {args.resolution}...")
        grid = ingest_las(args.file, args.resolution)
    else:
        print(f"Generating procedural terrain ({args.terrain_size}x{args.terrain_size})...")
        grid = generate_terrain(args.terrain_size)
    elapsed = time.perf_counter() - t0

    patch_count = sum(len(d) for d in grid.patch_lods.values())
    layer_count = len(grid.patch_lods)
    mem_bytes = grid.memory_bytes()
    if mem_bytes >= 1024 * 1024:
        mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
    else:
        mem_str = f"{mem_bytes / 1024:.1f} KB"

    total_vol, solid = grid.voxel_counts()

    def _fmt_count(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f} billion"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f} million"
        if n >= 1_000:
            return f"{n / 1_000:.0f}k"
        return str(n)

    print(f"Ready in {elapsed:.2f}s — {patch_count} patches across {layer_count} layer(s), "
          f"{_fmt_count(solid)} solid / {_fmt_count(total_vol)} total voxels, "
          f"dataset memory: {mem_str}")
    print(f"Serving on port {args.port}")
    asyncio.run(start_server(grid, "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
