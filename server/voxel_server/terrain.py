from __future__ import annotations

import numpy as np

from .noise import PerlinNoise
from .voxelgrid import VoxelGrid


def generate_terrain(size: int = 1896, height: int = 60, seed: int = 42) -> VoxelGrid:
    """Generate a hilly terrain as a VoxelGrid using Perlin FBM noise.

    Default parameters match the frontend demo (demo/main.ts):
    - 632x632 grid at 1.0m resolution ≈ 632m world extent
    - Perlin FBM: frequency=0.006, octaves=6, persistence=0.4
    - Height scale: 12m
    """
    noise = PerlinNoise(seed)

    # Build coordinate grids in world space (meters)
    # Center at origin like the frontend does
    half = size / 2
    xs = np.arange(size, dtype=np.float64) - half
    zs = np.arange(size, dtype=np.float64) - half
    wx, wz = np.meshgrid(xs, zs)

    # Generate heightmap using FBM with demo parameters
    heightmap = noise.fbm2d(
        wx, wz,
        frequency=0.006,
        octaves=6,
        persistence=0.4,
    )

    # Scale to height range: fbm returns ~[-1,1], map to [0, height-1]
    heightmap = (heightmap + 1) * 0.5  # [0, 1]
    heightmap = np.clip(heightmap * (height - 1), 0, height - 1).astype(np.int32)

    # Build palette: green(low) → brown(mid) → white(high)
    palette: list[tuple[int, int, int]] = []
    for i in range(height):
        t = i / max(height - 1, 1)
        if t < 0.4:
            g = int(80 + t / 0.4 * 100)
            palette.append((40, g, 30))
        elif t < 0.75:
            tt = (t - 0.4) / 0.35
            r = int(100 + tt * 55)
            g = int(80 + tt * 20)
            palette.append((r, g, 50))
        else:
            tt = (t - 0.75) / 0.25
            v = int(180 + tt * 75)
            palette.append((v, v, v))

    if not palette:
        palette.append((128, 128, 128))

    grid = VoxelGrid(resolution=1.0)
    grid.palette = palette

    # Fill voxel patches (vectorized per-patch)
    patches_x = (size + 63) // 64
    patches_z = (size + 63) // 64
    patches_y = (height + 63) // 64

    for pz in range(patches_z):
        z0 = pz * 64
        z1 = min(z0 + 64, size)
        for px in range(patches_x):
            x0 = px * 64
            x1 = min(x0 + 64, size)

            # Extract the heightmap slice for this patch's XZ footprint
            patch_heights = heightmap[z0:z1, x0:x1]  # shape (dz, dx)

            for py in range(patches_y):
                y0 = py * 64
                y1 = min(y0 + 64, height)
                if y0 > patch_heights.max():
                    continue

                patch = np.zeros((64, 64, 64), dtype=np.uint8)
                dx = x1 - x0
                dz = z1 - z0
                dy = y1 - y0

                # Create Y coordinate grid for this patch
                ys = np.arange(y0, y1, dtype=np.int32)  # (dy,)

                # For each (lx, lz), fill columns where wy <= h
                # heights_2d shape: (dz, dx) -> broadcast with ys (dy,)
                # mask shape: (dy, dz, dx) — True where y <= height at that column
                mask = ys[:, None, None] <= patch_heights[None, :, :]  # (dy, dz, dx)

                # Color indices: based on world Y, 1-indexed
                color_indices = np.minimum(ys, len(palette) - 1) + 1  # (dy,)
                # Broadcast to fill: patch axes are [lx, ly, lz]
                # mask is [ly, lz, lx] -> need to transpose to [lx, ly, lz]
                colors = np.where(mask, color_indices[:, None, None], 0).astype(np.uint8)
                # colors shape: (dy, dz, dx)
                # patch indexing: patch[lx, ly, lz] so we need (dx, dy, dz)
                patch[:dx, :dy, :dz] = colors.transpose(2, 0, 1)

                if patch.any():
                    grid.add_patch(0, px, py, pz, patch)

    return grid
