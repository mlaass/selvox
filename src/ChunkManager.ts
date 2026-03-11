import type { VoxelRenderer } from './VoxelRenderer.js';
import type { IVoxelDataSource } from './types.js';

export interface ChunkManagerOptions {
  radiusScale?: number;
  maxChunksPerFrame?: number;
  maxLodLevel?: number;
  chunkSize?: number;
  yExtent?: number;
}

interface ManagedChunk {
  chunkId: string;
  lodLevel: number;
  gridX: number;
  gridZ: number;
}

const DEFAULT_RADIUS_SCALE = 48;
const DEFAULT_MAX_CHUNKS_PER_FRAME = 4;
const DEFAULT_MAX_LOD = 6;
const BASE_CHUNK_SIZE = 16;

export class ChunkManager {
  private renderer: VoxelRenderer;
  private source: IVoxelDataSource;
  private radiusScale: number;
  private maxChunksPerFrame: number;
  private maxLodLevel: number;
  private chunkSize: number;
  private yExtent: number;

  private managedChunks = new Map<string, ManagedChunk>();
  private pendingLoads = new Set<string>();

  // Dirty-check: last camera grid cell (LOD 0 grid)
  private lastCamGridX = NaN;
  private lastCamGridZ = NaN;

  // Async overlap guard
  private updating = false;

  // True when all desired chunks are loaded (safe to skip update)
  private allDesiredLoaded = false;

  // Loaded voxels counter for the most recent update cycle
  private _lastLoadedVoxels = 0;

  constructor(
    renderer: VoxelRenderer,
    source: IVoxelDataSource,
    opts?: ChunkManagerOptions,
  ) {
    this.renderer = renderer;
    this.source = source;
    this.radiusScale = opts?.radiusScale ?? DEFAULT_RADIUS_SCALE;
    this.maxChunksPerFrame = opts?.maxChunksPerFrame ?? DEFAULT_MAX_CHUNKS_PER_FRAME;
    this.maxLodLevel = opts?.maxLodLevel ?? DEFAULT_MAX_LOD;
    this.chunkSize = opts?.chunkSize ?? BASE_CHUNK_SIZE;
    this.yExtent = opts?.yExtent ?? 60;
  }

  async initialize(): Promise<void> {
    // No pre-computation needed for infinite terrain
  }

  async update(cameraPosition: Float64Array): Promise<void> {
    // Guard against overlapping async calls
    if (this.updating) return;

    const camX = cameraPosition[0];
    const camZ = cameraPosition[2];

    // Dirty-check: only recompute when camera crosses a LOD 0 grid boundary
    const camGridX = Math.floor(camX / this.chunkSize);
    const camGridZ = Math.floor(camZ / this.chunkSize);

    if (camGridX === this.lastCamGridX && camGridZ === this.lastCamGridZ && this.allDesiredLoaded) {
      return;
    }

    this.updating = true;
    this._lastLoadedVoxels = 0;

    try {
      this.lastCamGridX = camGridX;
      this.lastCamGridZ = camGridZ;

      // Build the desired set from finest (LOD 0) to coarsest, using sub-cell coverage
      // to prevent overlap: a coarser chunk is only added if not all its finer sub-cells exist.
      //
      // Key format per LOD level: `L_gx_gz`
      const desiredByLod: Map<string, { gridX: number; gridZ: number; dist: number }>[] = [];
      for (let L = 0; L <= this.maxLodLevel; L++) {
        desiredByLod.push(new Map());
      }

      for (let L = 0; L <= this.maxLodLevel; L++) {
        const chunkSize = this.chunkSize * (1 << L);
        const outerRadius = this.radiusScale * (1 << L);
        const innerRadius = L > 0 ? this.radiusScale * (1 << (L - 1)) : 0;

        const gridCX = Math.floor(camX / chunkSize);
        const gridCZ = Math.floor(camZ / chunkSize);
        const scanRadius = Math.ceil(outerRadius / chunkSize) + 1;

        for (let gx = gridCX - scanRadius; gx <= gridCX + scanRadius; gx++) {
          for (let gz = gridCZ - scanRadius; gz <= gridCZ + scanRadius; gz++) {
            const cx = (gx + 0.5) * chunkSize;
            const cz = (gz + 0.5) * chunkSize;
            const dx = cx - camX;
            const dz = cz - camZ;
            const dist = Math.sqrt(dx * dx + dz * dz);

            if (dist >= innerRadius && dist < outerRadius) {
              // For LOD > 0, check if all 4 sub-cells at LOD L-1 are already covered
              if (L > 0) {
                const finerMap = desiredByLod[L - 1];
                const sx = gx * 2;
                const sz = gz * 2;
                const allCovered =
                  finerMap.has(`${sx}_${sz}`) &&
                  finerMap.has(`${sx + 1}_${sz}`) &&
                  finerMap.has(`${sx}_${sz + 1}`) &&
                  finerMap.has(`${sx + 1}_${sz + 1}`);
                if (allCovered) continue;
              }

              desiredByLod[L].set(`${gx}_${gz}`, { gridX: gx, gridZ: gz, dist });
            }
          }
        }
      }

      // Flatten into a single desired map keyed by the chunk key used for management
      const desired = new Map<string, { lodLevel: number; gridX: number; gridZ: number; dist: number }>();
      for (let L = 0; L <= this.maxLodLevel; L++) {
        for (const [cellKey, info] of desiredByLod[L]) {
          const key = `terrain_L${L}_${cellKey}`;
          desired.set(key, { lodLevel: L, gridX: info.gridX, gridZ: info.gridZ, dist: info.dist });
        }
      }

      // Unload chunks no longer desired (with hysteresis)
      const hysteresis = this.radiusScale * 0.15;
      const toUnload: string[] = [];
      for (const [key, managed] of this.managedChunks) {
        if (!desired.has(key)) {
          const chunkSize = this.chunkSize * (1 << managed.lodLevel);
          const cx = (managed.gridX + 0.5) * chunkSize;
          const cz = (managed.gridZ + 0.5) * chunkSize;
          const dx = cx - camX;
          const dz = cz - camZ;
          const dist = Math.sqrt(dx * dx + dz * dz);

          const outerRadius = this.radiusScale * (1 << managed.lodLevel);

          if (dist >= outerRadius + hysteresis) {
            toUnload.push(key);
          }
        }
      }

      for (const key of toUnload) {
        const managed = this.managedChunks.get(key)!;
        this.renderer.unloadChunk(managed.chunkId);
        this.managedChunks.delete(key);
      }

      // Collect new chunks to load, sorted by distance (closest first)
      const toLoad: { key: string; lodLevel: number; gridX: number; gridZ: number; dist: number }[] = [];
      for (const [key, info] of desired) {
        if (!this.managedChunks.has(key) && !this.pendingLoads.has(key)) {
          toLoad.push({ key, ...info });
        }
      }

      this.allDesiredLoaded = toLoad.length === 0;

      toLoad.sort((a, b) => a.dist - b.dist);

      const loadBatch = toLoad.slice(0, this.maxChunksPerFrame);

      const loadPromises = loadBatch.map(async (item) => {
        this.pendingLoads.add(item.key);

        const chunkSize = this.chunkSize * (1 << item.lodLevel);
        const yExtent = this.yExtent;
        const minX = item.gridX * chunkSize;
        const minZ = item.gridZ * chunkSize;
        const bbox = new Float64Array([
          minX, -yExtent, minZ,
          minX + chunkSize, yExtent, minZ + chunkSize,
        ]);

        try {
          const chunk = await this.source.requestChunk(bbox, item.lodLevel, 0);
          const voxelCount = chunk.data.byteLength / 64;

          if (voxelCount > 0) {
            this.renderer.loadChunk(
              chunk.id,
              chunk.data,
              voxelCount,
              chunk.worldPosition,
              chunk.lodLevel,
            );
            this._lastLoadedVoxels += voxelCount;
          }

          this.managedChunks.set(item.key, {
            chunkId: chunk.id,
            lodLevel: item.lodLevel,
            gridX: item.gridX,
            gridZ: item.gridZ,
          });
        } finally {
          this.pendingLoads.delete(item.key);
        }
      });

      await Promise.all(loadPromises);
    } finally {
      this.updating = false;
    }
  }

  get loadedChunkCount(): number {
    return this.managedChunks.size;
  }

  get lastLoadedVoxels(): number {
    return this._lastLoadedVoxels;
  }

  getActiveLodLevels(): Set<number> {
    const levels = new Set<number>();
    for (const chunk of this.managedChunks.values()) {
      levels.add(chunk.lodLevel);
    }
    return levels;
  }

  unloadAll(): void {
    for (const managed of this.managedChunks.values()) {
      this.renderer.unloadChunk(managed.chunkId);
    }
    this.managedChunks.clear();
    // Reset dirty-check so next update re-evaluates
    this.lastCamGridX = NaN;
    this.lastCamGridZ = NaN;
    this.allDesiredLoaded = false;
  }
}
