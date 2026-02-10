import type { IVoxelDataSource, VoxelDataChunk } from './types.js';

const VOXEL_STRIDE = 64; // bytes per voxel

// Pack RGBA color as u32 (little-endian: ABGR byte order)
function packColor(r: number, g: number, b: number): number {
  return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | (0xff << 24);
}

// World origin at a large UTM-scale offset to exercise RTE precision
const WORLD_CENTER = new Float64Array([500000, 0, 5500000]);
const CHUNK_SIZE = 4; // world-units per chunk side

interface LodConfig {
  gridN: number;
  voxelSize: number;
}

const LOD_CONFIGS: LodConfig[] = [
  { gridN: 16, voxelSize: 0.25 }, // LOD 0: 16^3 = 4096
  { gridN: 8,  voxelSize: 0.5 },  // LOD 1: 8^3 = 512
  { gridN: 4,  voxelSize: 1.0 },  // LOD 2: 4^3 = 64
  { gridN: 2,  voxelSize: 2.0 },  // LOD 3: 2^3 = 8
  { gridN: 1,  voxelSize: 4.0 },  // LOD 4: 1
  { gridN: 1,  voxelSize: 8.0 },  // LOD 5: 1
  { gridN: 1,  voxelSize: 16.0 }, // LOD 6: 1
];

function buildProceduralChunk(
  chunkOrigin: Float64Array,
  lodLevel: number,
): { data: ArrayBuffer; count: number } {
  const lod = LOD_CONFIGS[Math.min(lodLevel, LOD_CONFIGS.length - 1)];
  const N = lod.gridN;
  const count = N * N * N;
  const data = new ArrayBuffer(count * VOXEL_STRIDE);

  let idx = 0;
  for (let y = 0; y < N; y++) {
    // Height gradient color, shifted slightly per LOD for visual distinction
    const heightT = N > 1 ? y / (N - 1) : 0.5;
    const lodShift = lodLevel * 30;
    const r = Math.round(Math.min(255, 40 + (1 - heightT) * 120 + lodShift));
    const g = Math.round(Math.min(255, 160 - heightT * 60));
    const b = Math.round(Math.min(255, 80 + heightT * 175 + lodShift * 0.5));
    const color = packColor(r, g, b);

    for (let x = 0; x < N; x++) {
      for (let z = 0; z < N; z++) {
        // Positions in chunk-local space, centered in the chunk
        const px = (x + 0.5) * lod.voxelSize - CHUNK_SIZE * 0.5;
        const py = (y + 0.5) * lod.voxelSize - CHUNK_SIZE * 0.5;
        const pz = (z + 0.5) * lod.voxelSize - CHUNK_SIZE * 0.5;

        const offset = idx * VOXEL_STRIDE;
        const f32 = new Float32Array(data, offset, 16);
        const u32 = new Uint32Array(data, offset, 16);

        // State A = State B (no animation for mock data)
        f32[0] = px; f32[1] = py; f32[2] = pz; f32[3] = 0;
        u32[4] = color;
        f32[5] = lod.voxelSize * 0.95; // slight gap between voxels
        f32[6] = 0; f32[7] = 0;

        f32[8] = px; f32[9] = py; f32[10] = pz; f32[11] = 0;
        u32[12] = color;
        f32[13] = lod.voxelSize * 0.95;
        f32[14] = 0; f32[15] = 0;

        idx++;
      }
    }
  }

  return { data, count };
}

export class MockOctreeSource implements IVoxelDataSource {
  private gridExtent: number;

  constructor(gridExtent: number = 3) {
    this.gridExtent = gridExtent;
  }

  async getMetadata(): Promise<{
    worldBounds: Float64Array;
    maxLodDepth: number;
    coordinateSystem: 'cartesian' | 'geospatial';
  }> {
    const halfExtent = this.gridExtent * CHUNK_SIZE * 0.5;
    return {
      worldBounds: new Float64Array([
        WORLD_CENTER[0] - halfExtent, WORLD_CENTER[1] - halfExtent, WORLD_CENTER[2] - halfExtent,
        WORLD_CENTER[0] + halfExtent, WORLD_CENTER[1] + halfExtent, WORLD_CENTER[2] + halfExtent,
      ]),
      maxLodDepth: 6,
      coordinateSystem: 'geospatial',
    };
  }

  async requestChunk(
    bbox: Float64Array,
    lodLevel: number,
    _timeIndex: number,
  ): Promise<VoxelDataChunk> {
    // Derive chunk origin from bbox center
    const originX = (bbox[0] + bbox[3]) * 0.5;
    const originY = (bbox[1] + bbox[4]) * 0.5;
    const originZ = (bbox[2] + bbox[5]) * 0.5;
    const worldPosition = new Float64Array([originX, originY, originZ]);

    const { data, count: _count } = buildProceduralChunk(worldPosition, lodLevel);

    const id = `chunk_${originX.toFixed(0)}_${originY.toFixed(0)}_${originZ.toFixed(0)}_lod${lodLevel}`;

    return {
      id,
      lodLevel,
      worldPosition,
      data,
    };
  }

  /** Expose the grid layout for the ChunkManager */
  getChunkPositions(): Float64Array[] {
    const positions: Float64Array[] = [];
    const half = Math.floor(this.gridExtent / 2);
    for (let x = -half; x <= half; x++) {
      for (let y = -half; y <= half; y++) {
        for (let z = -half; z <= half; z++) {
          positions.push(new Float64Array([
            WORLD_CENTER[0] + x * CHUNK_SIZE,
            WORLD_CENTER[1] + y * CHUNK_SIZE,
            WORLD_CENTER[2] + z * CHUNK_SIZE,
          ]));
        }
      }
    }
    return positions;
  }
}
