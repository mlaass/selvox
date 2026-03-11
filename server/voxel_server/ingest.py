from __future__ import annotations

import numpy as np

from .voxelgrid import VoxelGrid


def ingest_las(path: str, resolution: float = 1.0) -> VoxelGrid:
    """Load a LAS/LAZ file and voxelize into a VoxelGrid."""
    import laspy

    las = laspy.read(path)
    xyz = las.xyz  # (N, 3) float64

    # Quantize positions to voxel coordinates
    origin = xyz.min(axis=0)
    voxel_coords = ((xyz - origin) / resolution).astype(np.int32)

    # Extract or generate colors
    has_color = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")
    if has_color:
        # LAS colors are often 16-bit, normalize to 8-bit
        red = np.asarray(las.red, dtype=np.float64)
        green = np.asarray(las.green, dtype=np.float64)
        blue = np.asarray(las.blue, dtype=np.float64)
        max_val = max(red.max(), green.max(), blue.max(), 1.0)
        if max_val > 255:
            red = (red / max_val * 255).astype(np.uint8)
            green = (green / max_val * 255).astype(np.uint8)
            blue = (blue / max_val * 255).astype(np.uint8)
        else:
            red = red.astype(np.uint8)
            green = green.astype(np.uint8)
            blue = blue.astype(np.uint8)
        colors = np.stack([red, green, blue], axis=1)  # (N, 3)
    else:
        # Height-based coloring
        heights = voxel_coords[:, 1].astype(np.float64)
        h_min, h_max = heights.min(), heights.max()
        t = (heights - h_min) / max(h_max - h_min, 1.0)
        colors = np.zeros((len(t), 3), dtype=np.uint8)
        colors[:, 0] = (40 + t * 180).astype(np.uint8)
        colors[:, 1] = (80 + t * 140).astype(np.uint8)
        colors[:, 2] = (30 + t * 200).astype(np.uint8)

    # Build palette via uniform quantization (6x6x7 = 252 bins)
    r_bins, g_bins, b_bins = 6, 6, 7
    ri = (colors[:, 0].astype(np.int32) * r_bins // 256).clip(0, r_bins - 1)
    gi = (colors[:, 1].astype(np.int32) * g_bins // 256).clip(0, g_bins - 1)
    bi = (colors[:, 2].astype(np.int32) * b_bins // 256).clip(0, b_bins - 1)
    palette_idx = ri * g_bins * b_bins + gi * b_bins + bi  # 0..251

    # Build actual palette colors (center of each bin)
    palette: list[tuple[int, int, int]] = []
    used_indices = np.unique(palette_idx)
    # Map from quantized index to 1-based palette index
    index_map = np.zeros(r_bins * g_bins * b_bins, dtype=np.uint8)
    for new_idx, old_idx in enumerate(used_indices):
        if new_idx >= 255:
            break
        r = int((old_idx // (g_bins * b_bins)) * 256 // r_bins + 128 // r_bins)
        g = int((old_idx % (g_bins * b_bins) // b_bins) * 256 // g_bins + 128 // g_bins)
        b = int((old_idx % b_bins) * 256 // b_bins + 128 // b_bins)
        palette.append((min(r, 255), min(g, 255), min(b, 255)))
        index_map[old_idx] = new_idx + 1  # 1-based

    # Map each point to its palette index
    voxel_palette = index_map[palette_idx]  # (N,) uint8, 1-based

    grid = VoxelGrid(resolution=resolution)
    grid.palette = palette

    # Group into 64x64x64 patches using a hash key for O(N) grouping
    patch_coords = voxel_coords // 64  # (N, 3)
    local_coords = voxel_coords % 64  # (N, 3)

    # Encode patch coords into a single int for fast grouping
    pc_max = patch_coords.max(axis=0) + 1
    patch_keys_flat = (
        patch_coords[:, 0].astype(np.int64) * pc_max[1] * pc_max[2]
        + patch_coords[:, 1].astype(np.int64) * pc_max[2]
        + patch_coords[:, 2].astype(np.int64)
    )

    # Sort by patch key for group-by
    order = np.argsort(patch_keys_flat, kind="mergesort")
    sorted_keys = patch_keys_flat[order]
    sorted_local = local_coords[order]
    sorted_palette = voxel_palette[order]
    sorted_patch_coords = patch_coords[order]

    # Find group boundaries
    breaks = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(sorted_keys)]])

    for start, end in zip(starts, ends):
        lc = sorted_local[start:end]
        pal = sorted_palette[start:end]
        pc = sorted_patch_coords[start]

        patch = np.zeros((64, 64, 64), dtype=np.uint8)
        # Clip to patch bounds (should already be valid, but be safe)
        valid = np.all((lc >= 0) & (lc < 64), axis=1)
        lc = lc[valid]
        pal = pal[valid]
        if len(lc) == 0:
            continue
        # Last-write-wins
        patch[lc[:, 0], lc[:, 1], lc[:, 2]] = pal
        grid.add_patch(int(pc[0]), int(pc[1]), int(pc[2]), patch)

    return grid
