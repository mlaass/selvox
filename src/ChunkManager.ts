import type { VoxelRenderer } from './VoxelRenderer.js';
import type { MockOctreeSource } from './MockOctreeSource.js';

const DEFAULT_LOD_THRESHOLDS = [20, 50, 100, 200, 400, 800];

interface ManagedChunk {
  position: Float64Array;
  currentLod: number;
  chunkId: string;
}

export class ChunkManager {
  private renderer: VoxelRenderer;
  private source: MockOctreeSource;
  private lodThresholds: number[];
  private managedChunks = new Map<string, ManagedChunk>();
  private chunkPositions: Float64Array[] = [];
  private maxDistance: number;

  constructor(
    renderer: VoxelRenderer,
    source: MockOctreeSource,
    lodThresholds?: number[],
  ) {
    this.renderer = renderer;
    this.source = source;
    this.lodThresholds = lodThresholds ?? DEFAULT_LOD_THRESHOLDS;
    this.maxDistance = this.lodThresholds[this.lodThresholds.length - 1] * 1.5;
  }

  async initialize(): Promise<void> {
    this.chunkPositions = this.source.getChunkPositions();
  }

  private positionKey(pos: Float64Array): string {
    return `${pos[0].toFixed(0)}_${pos[1].toFixed(0)}_${pos[2].toFixed(0)}`;
  }

  private distanceToChunk(cameraPos: Float64Array, chunkPos: Float64Array): number {
    const dx = cameraPos[0] - chunkPos[0];
    const dy = cameraPos[1] - chunkPos[1];
    const dz = cameraPos[2] - chunkPos[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private lodForDistance(distance: number): number {
    for (let i = 0; i < this.lodThresholds.length; i++) {
      if (distance < this.lodThresholds[i]) return i;
    }
    return this.lodThresholds.length;
  }

  async update(cameraPositionHigh: Float64Array): Promise<void> {
    const toUnload: string[] = [];

    // Check each known chunk position
    for (const pos of this.chunkPositions) {
      const key = this.positionKey(pos);
      const distance = this.distanceToChunk(cameraPositionHigh, pos);
      const desiredLod = this.lodForDistance(distance);

      if (distance > this.maxDistance) {
        // Too far: unload if loaded
        if (this.managedChunks.has(key)) {
          toUnload.push(key);
        }
        continue;
      }

      const existing = this.managedChunks.get(key);
      if (existing && existing.currentLod === desiredLod) {
        continue; // Already at correct LOD
      }

      // Need to load or change LOD
      const halfSize = 2; // CHUNK_SIZE / 2
      const bbox = new Float64Array([
        pos[0] - halfSize, pos[1] - halfSize, pos[2] - halfSize,
        pos[0] + halfSize, pos[1] + halfSize, pos[2] + halfSize,
      ]);

      const chunk = await this.source.requestChunk(bbox, desiredLod, 0);
      const voxelCount = chunk.data.byteLength / 64;

      this.renderer.loadChunk(
        chunk.id,
        chunk.data,
        voxelCount,
        chunk.worldPosition,
        chunk.lodLevel,
      );

      // Unload old chunk if LOD changed
      if (existing) {
        this.renderer.unloadChunk(existing.chunkId);
      }

      this.managedChunks.set(key, {
        position: pos,
        currentLod: desiredLod,
        chunkId: chunk.id,
      });
    }

    // Unload distant chunks
    for (const key of toUnload) {
      const managed = this.managedChunks.get(key)!;
      this.renderer.unloadChunk(managed.chunkId);
      this.managedChunks.delete(key);
    }
  }

  get loadedChunkCount(): number {
    return this.managedChunks.size;
  }

  getActiveLodLevels(): Set<number> {
    const levels = new Set<number>();
    for (const chunk of this.managedChunks.values()) {
      levels.add(chunk.currentLod);
    }
    return levels;
  }

  unloadAll(): void {
    for (const managed of this.managedChunks.values()) {
      this.renderer.unloadChunk(managed.chunkId);
    }
    this.managedChunks.clear();
  }
}
