from __future__ import annotations

import numpy as np


class VoxelGrid:
    """Sparse patch grid with precomputed LOD pyramids, keyed by layer."""

    def __init__(self, resolution: float = 1.0) -> None:
        self.patch_lods: dict[int, dict[tuple[int, int, int], list[np.ndarray]]] = {}
        self.palette: list[tuple[int, int, int]] = []
        self.resolution = resolution
        # Tracked as max patch coord + 1 in each axis, times 64
        self._max_coords = [0, 0, 0]

    @property
    def world_size(self) -> tuple[int, int, int]:
        return (self._max_coords[0], self._max_coords[1], self._max_coords[2])

    def add_patch(self, layer: int, px: int, py: int, pz: int, data: np.ndarray) -> None:
        """Store a 64x64x64 uint8 array and precompute 6 coarser LOD levels."""
        assert data.shape == (64, 64, 64) and data.dtype == np.uint8

        lods: list[np.ndarray] = [np.empty(0)] * 7  # placeholders
        lods[6] = data
        current = data
        for level in range(5, -1, -1):
            current = self._downsample(current)
            lods[level] = current

        layer_dict = self.patch_lods.setdefault(layer, {})
        layer_dict[(px, py, pz)] = lods
        self._max_coords[0] = max(self._max_coords[0], (px + 1) * 64)
        self._max_coords[1] = max(self._max_coords[1], (py + 1) * 64)
        self._max_coords[2] = max(self._max_coords[2], (pz + 1) * 64)

    def get_patch(self, layer: int, px: int, py: int, pz: int, level: int) -> np.ndarray | None:
        """Return precomputed LOD array, or None if patch is empty."""
        layer_dict = self.patch_lods.get(layer)
        if layer_dict is None:
            return None
        lods = layer_dict.get((px, py, pz))
        if lods is None:
            return None
        return lods[level]

    def list_patches(self, layer: int) -> list[tuple[int, int, int]]:
        """Return all non-empty patch coordinates in the given layer."""
        layer_dict = self.patch_lods.get(layer)
        if layer_dict is None:
            return []
        return list(layer_dict.keys())

    def delete_patch(self, layer: int, px: int, py: int, pz: int) -> None:
        """Remove a patch and all its LOD levels from the given layer."""
        layer_dict = self.patch_lods.get(layer)
        if layer_dict is not None:
            layer_dict.pop((px, py, pz), None)
            if not layer_dict:
                del self.patch_lods[layer]

    def list_layers(self) -> list[int]:
        """Return all layers that currently contain data."""
        return sorted(self.patch_lods.keys())

    def voxel_counts(self) -> tuple[int, int]:
        """Return (total_volume, solid_voxels) across all LOD-6 (full-res) patches."""
        total = 0
        solid = 0
        for layer_dict in self.patch_lods.values():
            for lods in layer_dict.values():
                data = lods[6]  # full-resolution patch
                total += data.size
                solid += int(np.count_nonzero(data))
        return total, solid

    def memory_bytes(self) -> int:
        """Total bytes used by all stored numpy arrays."""
        total = 0
        for layer_dict in self.patch_lods.values():
            for lods in layer_dict.values():
                for arr in lods:
                    total += arr.nbytes
        return total

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
