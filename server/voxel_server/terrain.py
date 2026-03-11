from __future__ import annotations

import numpy as np

from .voxelgrid import VoxelGrid


def generate_terrain(size: int = 256, height: int = 64, seed: int = 42) -> VoxelGrid:
    """Generate a hilly terrain as a VoxelGrid using smoothed noise."""
    rng = np.random.default_rng(seed)

    # Generate coarse noise and upsample
    coarse_size = max(size // 16, 4)
    coarse = rng.random((coarse_size, coarse_size), dtype=np.float32)

    # Upsample with linear interpolation
    from_x = np.linspace(0, coarse_size - 1, size)
    from_z = np.linspace(0, coarse_size - 1, size)
    ix = from_x.astype(int).clip(0, coarse_size - 2)
    iz = from_z.astype(int).clip(0, coarse_size - 2)
    fx = (from_x - ix).astype(np.float32)
    fz = (from_z - iz).astype(np.float32)

    # Bilinear interpolation
    heightmap = np.empty((size, size), dtype=np.float32)
    for zi in range(size):
        for xi in range(size):
            x0, z0 = ix[xi], iz[zi]
            tx, tz = fx[xi], fz[zi]
            v = (
                coarse[z0, x0] * (1 - tx) * (1 - tz)
                + coarse[z0, x0 + 1] * tx * (1 - tz)
                + coarse[z0 + 1, x0] * (1 - tx) * tz
                + coarse[z0 + 1, x0 + 1] * tx * tz
            )
            heightmap[zi, xi] = v

    # Add a second octave for detail
    coarse2_size = max(size // 4, 4)
    coarse2 = rng.random((coarse2_size, coarse2_size), dtype=np.float32)
    from_x2 = np.linspace(0, coarse2_size - 1, size)
    from_z2 = np.linspace(0, coarse2_size - 1, size)
    ix2 = from_x2.astype(int).clip(0, coarse2_size - 2)
    iz2 = from_z2.astype(int).clip(0, coarse2_size - 2)
    fx2 = (from_x2 - ix2).astype(np.float32)
    fz2 = (from_z2 - iz2).astype(np.float32)

    for zi in range(size):
        for xi in range(size):
            x0, z0 = ix2[xi], iz2[zi]
            tx, tz = fx2[xi], fz2[zi]
            v = (
                coarse2[z0, x0] * (1 - tx) * (1 - tz)
                + coarse2[z0, x0 + 1] * tx * (1 - tz)
                + coarse2[z0 + 1, x0] * (1 - tx) * tz
                + coarse2[z0 + 1, x0 + 1] * tx * tz
            )
            heightmap[zi, xi] += v * 0.3

    # Normalize to height range
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    heightmap = (heightmap * (height - 1)).astype(int)

    # Build palette: green(low) → brown(mid) → white(high)
    palette: list[tuple[int, int, int]] = []
    for i in range(height):
        t = i / max(height - 1, 1)
        if t < 0.4:
            # Green
            g = int(80 + t / 0.4 * 100)
            palette.append((40, g, 30))
        elif t < 0.75:
            # Brown
            tt = (t - 0.4) / 0.35
            r = int(100 + tt * 55)
            g = int(80 + tt * 20)
            palette.append((r, g, 50))
        else:
            # White/grey
            tt = (t - 0.75) / 0.25
            v = int(180 + tt * 75)
            palette.append((v, v, v))

    # Pad palette to at least 1 entry
    if not palette:
        palette.append((128, 128, 128))

    grid = VoxelGrid(resolution=1.0)
    grid.palette = palette

    # Fill voxel patches
    patches_x = (size + 63) // 64
    patches_z = (size + 63) // 64
    patches_y = (height + 63) // 64

    for pz in range(patches_z):
        for px in range(patches_x):
            patch = np.zeros((64, 64, 64), dtype=np.uint8)
            has_data = False
            for py in range(patches_y):
                for lz in range(64):
                    wz = pz * 64 + lz
                    if wz >= size:
                        continue
                    for lx in range(64):
                        wx = px * 64 + lx
                        if wx >= size:
                            continue
                        h = heightmap[wz, wx]
                        for ly in range(64):
                            wy = py * 64 + ly
                            if wy <= h:
                                # Palette index is 1-based (0 = empty)
                                color_idx = min(wy, len(palette) - 1) + 1
                                patch[lx, ly, lz] = color_idx
                                has_data = True
                if has_data:
                    grid.add_patch(px, py, pz, patch)
                    patch = np.zeros((64, 64, 64), dtype=np.uint8)
                    has_data = False

    return grid
