import { BlockAllocator } from './BlockAllocator.js';

const VOXEL_STRIDE = 64; // bytes per voxel
const DEFAULT_MAX_VOXELS = 1_000_000;

export interface PoolChunkInfo {
  startSlot: number;
  voxelCount: number;
  worldOrigin: Float64Array;
  lodLevel: number;
  chunkIndex: number;
}

export class VoxelPool {
  private device: GPUDevice;
  private allocator: BlockAllocator;
  private _buffer: GPUBuffer;
  private chunks = new Map<string, PoolChunkInfo>();
  private nextChunkIndex = 0;
  private freeChunkIndices: number[] = [];

  constructor(device: GPUDevice, maxVoxels: number = DEFAULT_MAX_VOXELS) {
    this.device = device;
    this.allocator = new BlockAllocator(maxVoxels);
    this._buffer = device.createBuffer({
      size: maxVoxels * VOXEL_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  get buffer(): GPUBuffer {
    return this._buffer;
  }

  get totalVoxels(): number {
    let total = 0;
    for (const info of this.chunks.values()) {
      total += info.voxelCount;
    }
    return total;
  }

  get chunkCount(): number {
    return this.chunks.size;
  }

  private allocChunkIndex(): number {
    if (this.freeChunkIndices.length > 0) {
      return this.freeChunkIndices.pop()!;
    }
    return this.nextChunkIndex++;
  }

  private freeChunkIndex(index: number): void {
    this.freeChunkIndices.push(index);
  }

  loadChunk(
    id: string,
    data: ArrayBuffer,
    voxelCount: number,
    worldOrigin?: Float64Array,
    lodLevel?: number,
  ): void {
    // Replace semantics: unload existing chunk with same id
    if (this.chunks.has(id)) {
      this.unloadChunk(id);
    }

    const startSlot = this.allocator.alloc(voxelCount);
    if (startSlot === null) {
      throw new Error(`VoxelPool: out of memory — cannot allocate ${voxelCount} slots for chunk "${id}"`);
    }

    this.device.queue.writeBuffer(
      this._buffer,
      startSlot * VOXEL_STRIDE,
      data,
      0,
      voxelCount * VOXEL_STRIDE,
    );

    this.chunks.set(id, {
      startSlot,
      voxelCount,
      worldOrigin: worldOrigin ?? new Float64Array([0, 0, 0]),
      lodLevel: lodLevel ?? 0,
      chunkIndex: this.allocChunkIndex(),
    });
  }

  unloadChunk(id: string): void {
    const info = this.chunks.get(id);
    if (!info) return;
    this.allocator.free(info.startSlot, info.voxelCount);
    this.freeChunkIndex(info.chunkIndex);
    this.chunks.delete(id);
  }

  hasChunk(id: string): boolean {
    return this.chunks.has(id);
  }

  getChunkInfo(id: string): PoolChunkInfo | undefined {
    return this.chunks.get(id);
  }

  forEachChunk(cb: (firstInstance: number, instanceCount: number, chunkInfo: PoolChunkInfo) => void): void {
    for (const info of this.chunks.values()) {
      cb(info.startSlot, info.voxelCount, info);
    }
  }

  dispose(): void {
    this._buffer.destroy();
    this.chunks.clear();
  }
}
