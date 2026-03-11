from __future__ import annotations

import numpy as np


class VoxelGrid:
    """Sparse patch grid with precomputed LOD pyramids."""

    def __init__(self, resolution: float = 1.0) -> None:
        self.patch_lods: dict[tuple[int, int, int], list[np.ndarray]] = {}
        self.palette: list[tuple[int, int, int]] = []
        self.resolution = resolution
        # Tracked as max patch coord + 1 in each axis, times 64
        self._max_coords = [0, 0, 0]

    @property
    def world_size(self) -> tuple[int, int, int]:
        return (self._max_coords[0], self._max_coords[1], self._max_coords[2])

    def add_patch(self, px: int, py: int, pz: int, data: np.ndarray) -> None:
        """Store a 64x64x64 uint8 array and precompute 6 coarser LOD levels."""
        assert data.shape == (64, 64, 64) and data.dtype == np.uint8

        lods: list[np.ndarray] = [np.empty(0)] * 7  # placeholders
        lods[6] = data
        current = data
        for level in range(5, -1, -1):
            current = self._downsample(current)
            lods[level] = current

        self.patch_lods[(px, py, pz)] = lods
        self._max_coords[0] = max(self._max_coords[0], (px + 1) * 64)
        self._max_coords[1] = max(self._max_coords[1], (py + 1) * 64)
        self._max_coords[2] = max(self._max_coords[2], (pz + 1) * 64)

    def get_patch(self, px: int, py: int, pz: int, level: int) -> np.ndarray | None:
        """Return precomputed LOD array, or None if patch is empty."""
        lods = self.patch_lods.get((px, py, pz))
        if lods is None:
            return None
        return lods[level]

    def list_patches(self) -> list[tuple[int, int, int]]:
        """Return all non-empty patch coordinates."""
        return list(self.patch_lods.keys())

    @staticmethod
    def _downsample(data: np.ndarray) -> np.ndarray:
        """Downsample by 2x: take max non-zero value in each 2x2x2 block.

        Fast vectorized approach — not true majority vote, but deterministic
        and sufficient for palette-indexed voxels.
        """
        n = data.shape[0]
        if n <= 1:
            return data.copy()
        h = n // 2
        # Reshape into 2x2x2 blocks → (h, h, h, 8)
        blocks = data.reshape(h, 2, h, 2, h, 2)
        flat = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(h, h, h, 8)
        return flat.max(axis=3).astype(np.uint8)
