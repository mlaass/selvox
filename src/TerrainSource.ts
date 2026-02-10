import type { IVoxelDataSource, VoxelDataChunk } from './types.js';
import { PerlinNoise } from './noise.js';
import type { FbmOptions } from './noise.js';

const VOXEL_STRIDE = 64; // bytes per voxel
const GRID_N = 16; // voxels per chunk axis

function packColor(r: number, g: number, b: number): number {
  return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | (0xff << 24);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function colorForHeight(h: number, heightScale: number): number {
  // Normalize height into [0, 1]
  const t = clamp((h / heightScale + 1) * 0.5, 0, 1);

  // Green (low) → Brown (mid) → Gray (high) → White (peak)
  let r: number, g: number, b: number;
  if (t < 0.3) {
    // Green grass
    const s = t / 0.3;
    r = Math.round(40 + s * 60);
    g = Math.round(140 + s * 30);
    b = Math.round(30 + s * 20);
  } else if (t < 0.6) {
    // Brown dirt/rock
    const s = (t - 0.3) / 0.3;
    r = Math.round(100 + s * 40);
    g = Math.round(170 - s * 80);
    b = Math.round(50 - s * 10);
  } else if (t < 0.85) {
    // Gray rock
    const s = (t - 0.6) / 0.25;
    r = Math.round(140 + s * 40);
    g = Math.round(90 + s * 50);
    b = Math.round(40 + s * 80);
  } else {
    // White snow
    const s = (t - 0.85) / 0.15;
    r = Math.round(180 + s * 75);
    g = Math.round(140 + s * 115);
    b = Math.round(120 + s * 135);
  }

  return packColor(r, g, b);
}

export interface TerrainOptions {
  seed?: number;
  heightScale?: number;
  baseHeight?: number;
  surfaceDepth?: number;
  noiseOptions?: FbmOptions;
}

export class TerrainSource implements IVoxelDataSource {
  private noise: PerlinNoise;
  private heightScale: number;
  private baseHeight: number;
  private surfaceDepth: number;
  private noiseOptions: FbmOptions;

  constructor(opts?: TerrainOptions) {
    this.noise = new PerlinNoise(opts?.seed ?? 42);
    this.heightScale = opts?.heightScale ?? 40;
    this.baseHeight = opts?.baseHeight ?? 0;
    this.surfaceDepth = opts?.surfaceDepth ?? 4;
    this.noiseOptions = opts?.noiseOptions ?? {};
  }

  async getMetadata(): Promise<{
    worldBounds: Float64Array;
    maxLodDepth: number;
    coordinateSystem: 'cartesian' | 'geospatial';
  }> {
    // Infinite terrain — return very large bounds
    const extent = 1e6;
    return {
      worldBounds: new Float64Array([
        -extent, -this.heightScale * 2, -extent,
        extent, this.heightScale * 2, extent,
      ]),
      maxLodDepth: 6,
      coordinateSystem: 'cartesian',
    };
  }

  async requestChunk(
    bbox: Float64Array,
    lodLevel: number,
    _timeIndex: number,
  ): Promise<VoxelDataChunk> {
    const chunkSize = bbox[3] - bbox[0]; // width in X
    const voxelSize = chunkSize / GRID_N;
    const chunkMinX = bbox[0];
    const chunkMinZ = bbox[2];

    // Chunk origin (center of bbox)
    const originX = (bbox[0] + bbox[3]) * 0.5;
    const originZ = (bbox[2] + bbox[5]) * 0.5;

    // Quick height range check — sample noise at 5 points
    const cornerSamples = [
      this.sampleHeight(chunkMinX, chunkMinZ),
      this.sampleHeight(chunkMinX + chunkSize, chunkMinZ),
      this.sampleHeight(chunkMinX, chunkMinZ + chunkSize),
      this.sampleHeight(chunkMinX + chunkSize, chunkMinZ + chunkSize),
      this.sampleHeight(originX, originZ),
    ];
    const minH = Math.min(...cornerSamples) - this.surfaceDepth * voxelSize;
    const maxH = Math.max(...cornerSamples);

    // The Y extent this chunk covers
    const yExtent = this.heightScale * 1.5;
    const chunkMinY = this.baseHeight - yExtent;
    const chunkMaxY = this.baseHeight + yExtent;

    // The bbox Y range for the origin
    const originY = (chunkMinY + chunkMaxY) * 0.5;

    const gridX = Math.floor(originX / chunkSize);
    const gridZ = Math.floor(originZ / chunkSize);
    const id = `terrain_L${lodLevel}_${gridX}_${gridZ}`;

    // If terrain doesn't intersect this chunk's potential Y range, return empty
    if (minH > chunkMaxY || maxH < chunkMinY) {
      return {
        id,
        lodLevel,
        worldPosition: new Float64Array([originX, originY, originZ]),
        data: new ArrayBuffer(0),
      };
    }

    // Generate voxels — collect into temp array then pack
    const voxels: { px: number; py: number; pz: number; color: number }[] = [];

    for (let gx = 0; gx < GRID_N; gx++) {
      for (let gz = 0; gz < GRID_N; gz++) {
        const worldX = chunkMinX + (gx + 0.5) * voxelSize;
        const worldZ = chunkMinZ + (gz + 0.5) * voxelSize;
        const h = this.sampleHeight(worldX, worldZ);
        const color = colorForHeight(h, this.heightScale);

        // Local positions relative to chunk origin
        const localX = (gx + 0.5) * voxelSize - chunkSize * 0.5;
        const localZ = (gz + 0.5) * voxelSize - chunkSize * 0.5;

        for (let gy = 0; gy < GRID_N; gy++) {
          const worldY = chunkMinY + (gy + 0.5) * (chunkMaxY - chunkMinY) / GRID_N;

          if (worldY <= h && worldY >= h - this.surfaceDepth * voxelSize) {
            const localY = worldY - originY;
            voxels.push({ px: localX, py: localY, pz: localZ, color });
          }
        }
      }
    }

    const count = voxels.length;
    const data = new ArrayBuffer(count * VOXEL_STRIDE);
    const scale = voxelSize * 0.95;

    for (let i = 0; i < count; i++) {
      const v = voxels[i];
      const offset = i * VOXEL_STRIDE;
      const f32 = new Float32Array(data, offset, 16);
      const u32 = new Uint32Array(data, offset, 16);

      // State A
      f32[0] = v.px; f32[1] = v.py; f32[2] = v.pz; f32[3] = 0;
      u32[4] = v.color;
      f32[5] = scale;
      f32[6] = 0; f32[7] = 0;

      // State B = State A (no animation)
      f32[8] = v.px; f32[9] = v.py; f32[10] = v.pz; f32[11] = 0;
      u32[12] = v.color;
      f32[13] = scale;
      f32[14] = 0; f32[15] = 0;
    }

    return {
      id,
      lodLevel,
      worldPosition: new Float64Array([originX, originY, originZ]),
      data,
    };
  }

  private sampleHeight(x: number, z: number): number {
    return this.baseHeight + this.noise.fbm2D(x, z, this.noiseOptions) * this.heightScale;
  }
}
