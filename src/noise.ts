export interface FbmOptions {
  octaves?: number;
  persistence?: number;
  lacunarity?: number;
  frequency?: number;
  amplitude?: number;
}

// Quintic fade: 6t^5 - 15t^4 + 10t^3
function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

// 2D gradient vectors (8 directions)
const GRAD2_X = [1, -1, 1, -1, 1, -1, 0, 0];
const GRAD2_Z = [0, 0, 1, 1, -1, -1, 1, -1];

function grad2D(hash: number, x: number, z: number): number {
  const h = hash & 7;
  return GRAD2_X[h] * x + GRAD2_Z[h] * z;
}

export class PerlinNoise {
  private perm: Uint8Array;

  constructor(seed: number = 42) {
    // Build seeded permutation table
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;

    // Fisher-Yates shuffle with simple seedable PRNG (mulberry32)
    let s = seed | 0;
    for (let i = 255; i > 0; i--) {
      // mulberry32 step
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      const r = ((t ^ (t >>> 14)) >>> 0) % (i + 1);
      const tmp = p[i];
      p[i] = p[r];
      p[r] = tmp;
    }

    // Double the table to avoid index wrapping
    this.perm = new Uint8Array(512);
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
  }

  noise2D(x: number, z: number): number {
    const perm = this.perm;

    // Grid cell coordinates
    const xi = Math.floor(x) & 255;
    const zi = Math.floor(z) & 255;

    // Fractional position within cell
    const xf = x - Math.floor(x);
    const zf = z - Math.floor(z);

    // Fade curves
    const u = fade(xf);
    const v = fade(zf);

    // Hash corners
    const aa = perm[perm[xi] + zi];
    const ab = perm[perm[xi] + zi + 1];
    const ba = perm[perm[xi + 1] + zi];
    const bb = perm[perm[xi + 1] + zi + 1];

    // Gradient dot products at corners, bilinear interpolation
    return lerp(
      lerp(grad2D(aa, xf, zf), grad2D(ba, xf - 1, zf), u),
      lerp(grad2D(ab, xf, zf - 1), grad2D(bb, xf - 1, zf - 1), u),
      v,
    );
  }

  fbm2D(x: number, z: number, opts?: FbmOptions): number {
    const octaves = opts?.octaves ?? 6;
    const persistence = opts?.persistence ?? 0.5;
    const lacunarity = opts?.lacunarity ?? 2.0;
    let freq = opts?.frequency ?? 0.01;
    let amp = opts?.amplitude ?? 1.0;

    let sum = 0;
    let maxAmp = 0;

    for (let i = 0; i < octaves; i++) {
      sum += this.noise2D(x * freq, z * freq) * amp;
      maxAmp += amp;
      amp *= persistence;
      freq *= lacunarity;
    }

    return sum / maxAmp; // Normalize to [-1, 1]
  }
}
