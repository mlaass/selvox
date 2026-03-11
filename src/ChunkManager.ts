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

  // Throttled logging
  private lastLogTime = 0;

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

      // Build the desired set via top-down quadtree subdivision.
      // Start at the coarsest LOD covering the max view distance, then
      // recursively subdivide cells whose distance warrants a finer LOD.
      // This guarantees complete coverage: a cell either stays or splits
      // into 4 children — it never disappears.
      const desired = new Map<string, { lodLevel: number; gridX: number; gridZ: number; dist: number }>();
      const maxViewDist = this.radiusScale * (1 << this.maxLodLevel);
      const coarseCellSize = this.chunkSize * (1 << this.maxLodLevel);
      const coarseGridCX = Math.floor(camX / coarseCellSize);
      const coarseGridCZ = Math.floor(camZ / coarseCellSize);
      const coarseScan = Math.ceil(maxViewDist / coarseCellSize) + 1;

      for (let gx = coarseGridCX - coarseScan; gx <= coarseGridCX + coarseScan; gx++) {
        for (let gz = coarseGridCZ - coarseScan; gz <= coarseGridCZ + coarseScan; gz++) {
          const cx = (gx + 0.5) * coarseCellSize;
          const cz = (gz + 0.5) * coarseCellSize;
          const dist = Math.sqrt((cx - camX) ** 2 + (cz - camZ) ** 2);
          if (dist < maxViewDist + coarseCellSize) {
            this.buildDesired(camX, camZ, this.maxLodLevel, gx, gz, desired);
          }
        }
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
          console.debug(`[CM] load ${item.key} lod=${item.lodLevel} dist=${item.dist.toFixed(0)}`);
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

      // Unload stale chunks AFTER new ones are loaded — prevents gaps/pops.
      // Old coarse chunks remain visible while finer replacements load.
      const toUnload: string[] = [];
      for (const [key] of this.managedChunks) {
        if (!desired.has(key)) {
          toUnload.push(key);
        }
      }
      for (const key of toUnload) {
        const managed = this.managedChunks.get(key)!;
        console.debug(`[CM] unload ${key}`);
        this.renderer.unloadChunk(managed.chunkId);
        this.managedChunks.delete(key);
      }

      // Throttled summary log (max once per second)
      const now = performance.now();
      if (now - this.lastLogTime > 1000) {
        this.lastLogTime = now;
        const lodCounts = new Map<number, number>();
        for (const m of this.managedChunks.values()) {
          lodCounts.set(m.lodLevel, (lodCounts.get(m.lodLevel) ?? 0) + 1);
        }
        const desiredLodCounts = new Map<number, number>();
        for (const d of desired.values()) {
          desiredLodCounts.set(d.lodLevel, (desiredLodCounts.get(d.lodLevel) ?? 0) + 1);
        }
        const lodStr = [...lodCounts.entries()].sort((a, b) => a[0] - b[0]).map(([l, c]) => `L${l}:${c}`).join(' ');
        const desiredStr = [...desiredLodCounts.entries()].sort((a, b) => a[0] - b[0]).map(([l, c]) => `L${l}:${c}`).join(' ');
        console.debug(`[CM] managed=${this.managedChunks.size} [${lodStr}] desired=[${desiredStr}] +${loadBatch.length} -${toUnload.length}`);
      }
    } finally {
      this.updating = false;
    }
  }

  private lodForDistance(dist: number): number {
    if (dist <= this.radiusScale) return 0;
    const lod = Math.floor(Math.log2(dist / this.radiusScale));
    return Math.min(lod, this.maxLodLevel);
  }

  private buildDesired(
    camX: number,
    camZ: number,
    L: number,
    gx: number,
    gz: number,
    desired: Map<string, { lodLevel: number; gridX: number; gridZ: number; dist: number }>,
  ): void {
    const cellSize = this.chunkSize * (1 << L);
    const cx = (gx + 0.5) * cellSize;
    const cz = (gz + 0.5) * cellSize;
    const dist = Math.sqrt((cx - camX) ** 2 + (cz - camZ) ** 2);
    const wantedLod = this.lodForDistance(dist);

    if (wantedLod >= L || L === 0) {
      // Keep this cell at LOD L
      const key = `terrain_L${L}_${gx}_${gz}`;
      desired.set(key, { lodLevel: L, gridX: gx, gridZ: gz, dist });
    } else {
      // Subdivide into 4 children at L-1
      for (let dx = 0; dx < 2; dx++) {
        for (let dz = 0; dz < 2; dz++) {
          this.buildDesired(camX, camZ, L - 1, gx * 2 + dx, gz * 2 + dz, desired);
        }
      }
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
