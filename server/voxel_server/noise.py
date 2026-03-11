"""Vectorized 2D Perlin noise with FBM, matching the frontend PerlinNoise class."""

from __future__ import annotations

import numpy as np


# 8 gradient directions (matching frontend GRAD2_X/GRAD2_Z)
_GRAD2_X = np.array([1, -1, 1, -1, 1, -1, 0, 0], dtype=np.float64)
_GRAD2_Z = np.array([0, 0, 1, 1, -1, -1, 1, -1], dtype=np.float64)


def _build_perm(seed: int) -> np.ndarray:
    """Build a 512-entry permutation table using mulberry32 PRNG (matches frontend)."""
    p = np.arange(256, dtype=np.int32)

    # Fisher-Yates shuffle with mulberry32 (must match JS exactly)
    s = np.int32(seed)
    for i in range(255, 0, -1):
        # mulberry32 step (matching JS Math.imul and unsigned shifts)
        s = np.int32(np.int64(s) + np.int64(0x6D2B79F5))
        t = np.int64(s ^ (np.uint32(s) >> np.uint32(15))) * np.int64(1 | s)
        t = np.int64(np.int32(t))
        t2 = np.int64(np.int32(t ^ (np.uint32(np.int32(t)) >> np.uint32(7))))
        t2 = np.int64(np.int32(t2 * np.int64(61 | np.int32(t2)))) ^ t2
        r_val = int(np.uint32(np.int32(t2) ^ (np.uint32(np.int32(t2)) >> np.uint32(14))))
        r = r_val % (i + 1)
        p[i], p[r] = p[r], p[i]

    perm = np.empty(512, dtype=np.int32)
    perm[:256] = p
    perm[256:] = p
    return perm


def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic fade: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad2d(hashes: np.ndarray, dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Vectorized gradient dot product."""
    h = hashes & 7
    return _GRAD2_X[h] * dx + _GRAD2_Z[h] * dz


class PerlinNoise:
    """2D Perlin noise generator with seeded permutation table."""

    def __init__(self, seed: int = 42) -> None:
        self.perm = _build_perm(seed)

    def noise2d(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate 2D Perlin noise at arrays of (x, z) coordinates.

        x, z should be broadcastable numpy arrays. Returns array of noise values in ~[-1, 1].
        """
        perm = self.perm

        xi = np.floor(x).astype(np.int32) & 255
        zi = np.floor(z).astype(np.int32) & 255

        xf = x - np.floor(x)
        zf = z - np.floor(z)

        u = _fade(xf)
        v = _fade(zf)

        # Hash corners
        aa = perm[perm[xi] + zi]
        ab = perm[perm[xi] + zi + 1]
        ba = perm[perm[xi + 1] + zi]
        bb = perm[perm[xi + 1] + zi + 1]

        # Gradient dot products + bilinear interpolation
        x1 = _grad2d(aa, xf, zf) * (1 - u) + _grad2d(ba, xf - 1, zf) * u
        x2 = _grad2d(ab, xf, zf - 1) * (1 - u) + _grad2d(bb, xf - 1, zf - 1) * u

        return x1 * (1 - v) + x2 * v

    def fbm2d(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        frequency: float = 0.006,
        octaves: int = 6,
        persistence: float = 0.4,
        lacunarity: float = 2.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Fractal Brownian Motion using Perlin noise."""
        result = np.zeros_like(x, dtype=np.float64)
        max_amp = 0.0
        freq = frequency
        amp = amplitude

        for _ in range(octaves):
            result += self.noise2d(x * freq, z * freq) * amp
            max_amp += amp
            amp *= persistence
            freq *= lacunarity

        return result / max_amp
