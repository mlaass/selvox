import type { IVoxelDataSource, VoxelDataChunk } from '../types.js';
import type { WebSocketVoxelSourceOptions, ServerMetadata } from './types.js';
import {
  encodeHandshake,
  encodeRequestBatch,
  encodeRequestPatch,
  encodeListPatches,
  decodeMessage,
  type DecodedMessage,
  type PatchDataMessage,
  type BatchPatchDataMessage,
} from './codec.js';

const VOXEL_STRIDE = 64;
const DEFAULT_URL = 'ws://localhost:9876';
const DEFAULT_CONNECT_TIMEOUT = 5000;
const DEFAULT_REQUEST_TIMEOUT = 10000;

interface PendingRequest {
  resolve: (msg: DecodedMessage) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class WebSocketVoxelSource implements IVoxelDataSource {
  private url: string;
  private connectTimeout: number;
  private requestTimeout: number;
  private resolution: number;
  private layer: number;
  private ws: WebSocket | null = null;
  private metadata: ServerMetadata | null = null;
  private knownPatches = new Set<string>();
  private pending = new Map<number, PendingRequest>();
  private nextRequestId = 1;

  constructor(opts?: WebSocketVoxelSourceOptions) {
    this.url = opts?.url ?? DEFAULT_URL;
    this.connectTimeout = opts?.connectTimeout ?? DEFAULT_CONNECT_TIMEOUT;
    this.requestTimeout = opts?.requestTimeout ?? DEFAULT_REQUEST_TIMEOUT;
    this.resolution = opts?.resolution ?? 1.0;
    this.layer = opts?.layer ?? 0;
  }

  async connect(): Promise<void> {
    const ws = new WebSocket(this.url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connect timeout'));
      }, this.connectTimeout);

      ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error('WebSocket connection failed'));
      };
    });

    ws.onmessage = (ev: MessageEvent) => {
      const msg = decodeMessage(ev.data as ArrayBuffer);
      const pending = this.pending.get(msg.requestId);
      if (pending) {
        clearTimeout(pending.timer);
        this.pending.delete(msg.requestId);
        if (msg.type === 'error') {
          pending.reject(new Error(`Server error ${msg.code}: ${msg.message}`));
        } else {
          pending.resolve(msg);
        }
      }
    };

    ws.onclose = () => {
      for (const p of this.pending.values()) {
        clearTimeout(p.timer);
        p.reject(new Error('WebSocket closed'));
      }
      this.pending.clear();
    };

    ws.onerror = () => {
      for (const p of this.pending.values()) {
        clearTimeout(p.timer);
        p.reject(new Error('WebSocket error'));
      }
      this.pending.clear();
    };

    // Handshake
    const metaMsg = await this.sendRequest(encodeHandshake);
    if (metaMsg.type !== 'metadata') throw new Error('Expected metadata response');
    this.metadata = {
      worldSize: metaMsg.worldSize,
      maxLevel: metaMsg.maxLevel,
      palette: metaMsg.palette,
    };

    // Discover patches at coarsest level
    const listMsg = await this.sendRequest((id) => encodeListPatches(id, this.layer, 0));
    if (listMsg.type !== 'patch_list') throw new Error('Expected patch list response');
    this.knownPatches.clear();
    for (const c of listMsg.coords) {
      this.knownPatches.add(`${c.px}_${c.py}_${c.pz}`);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new Error('Disconnected'));
    }
    this.pending.clear();
    this.metadata = null;
    this.knownPatches.clear();
  }

  async getMetadata(): Promise<{
    worldBounds: Float64Array;
    maxLodDepth: number;
    coordinateSystem: 'cartesian' | 'geospatial';
  }> {
    if (!this.metadata) throw new Error('Not connected');
    const ws = this.metadata.worldSize;
    const res = this.resolution;
    return {
      worldBounds: new Float64Array([0, 0, 0, ws[0] * res, ws[1] * res, ws[2] * res]),
      maxLodDepth: this.metadata.maxLevel,
      coordinateSystem: 'cartesian',
    };
  }

  async requestChunk(
    bbox: Float64Array,
    lodLevel: number,
    _timeIndex: number,
  ): Promise<VoxelDataChunk> {
    if (!this.metadata) throw new Error('Not connected');

    const res = this.resolution;
    const patchWorld = 64 * res;

    // A chunk at lodLevel covers patchSpan × patchSpan base patches
    const patchSpan = 1 << lodLevel;
    const pxMin = Math.round(bbox[0] / patchWorld);
    const pzMin = Math.round(bbox[2] / patchWorld);

    const serverLevel = this.metadata.maxLevel - lodLevel;
    const maxPy = Math.ceil(this.metadata.worldSize[1] / 64) - 1;
    const minPy = 0;

    // Collect all (px, py, pz) patches that exist within the bbox footprint
    const allPatches: { px: number; py: number; pz: number }[] = [];
    for (let dpx = 0; dpx < patchSpan; dpx++) {
      for (let dpz = 0; dpz < patchSpan; dpz++) {
        const px = pxMin + dpx;
        const pz = pzMin + dpz;
        for (let py = minPy; py <= maxPy; py++) {
          if (this.knownPatches.has(`${px}_${py}_${pz}`)) {
            allPatches.push({ px, py, pz });
          }
        }
      }
    }

    const chunkWorldY = this.metadata.worldSize[1] * res * 0.5;
    const chunkCenterX = pxMin * patchWorld + patchSpan * patchWorld * 0.5;
    const chunkCenterZ = pzMin * patchWorld + patchSpan * patchWorld * 0.5;
    const id = `ws_L${lodLevel}_${this.layer}_${pxMin}_${pzMin}`;

    console.debug(`[WS] request lod=${lodLevel} srvLvl=${serverLevel} patches=${allPatches.length} bbox=[${bbox[0].toFixed(0)},${bbox[2].toFixed(0)}..${bbox[3].toFixed(0)},${bbox[5].toFixed(0)}]`);

    if (allPatches.length === 0) {
      return {
        id,
        lodLevel,
        worldPosition: new Float64Array([chunkCenterX, chunkWorldY, chunkCenterZ]),
        data: new ArrayBuffer(0),
      };
    }

    // Request patches
    let patchResults: { level: number; px: number; py: number; pz: number; encoding: number; voxelData: Uint8Array | null }[];

    if (allPatches.length === 1) {
      const { px, py, pz } = allPatches[0];
      const msg = await this.sendRequest((reqId) =>
        encodeRequestPatch(reqId, this.layer, serverLevel, px, py, pz),
      );
      if (msg.type !== 'patch_data') throw new Error('Expected patch data');
      const pd = msg as PatchDataMessage;
      patchResults = [{ level: pd.level, px: pd.px, py: pd.py, pz: pd.pz, encoding: pd.encoding, voxelData: pd.voxelData }];
    } else {
      const batchPatches = allPatches.map((p) => ({ level: serverLevel, px: p.px, py: p.py, pz: p.pz }));
      const msg = await this.sendRequest((reqId) =>
        encodeRequestBatch(reqId, this.layer, batchPatches),
      );
      if (msg.type !== 'batch_patch_data') throw new Error('Expected batch patch data');
      patchResults = (msg as BatchPatchDataMessage).patches;
    }

    // Expand patches to GPU format
    const buffers: ArrayBuffer[] = [];
    for (const patch of patchResults) {
      if (patch.encoding === 0 || !patch.voxelData) continue;
      const xOffset = (patch.px * patchWorld + patchWorld * 0.5) - chunkCenterX;
      const yOffset = (patch.py * patchWorld + patchWorld * 0.5) - chunkWorldY;
      const zOffset = (patch.pz * patchWorld + patchWorld * 0.5) - chunkCenterZ;
      const expanded = this.expandPatchToGPU(patch.voxelData, patch.level, xOffset, yOffset, zOffset);
      if (expanded.byteLength > 0) buffers.push(expanded);
    }

    console.debug(`[WS] expanded ${buffers.reduce((s, b) => s + b.byteLength / VOXEL_STRIDE, 0)} voxels from ${patchResults.length} patches`);

    // Concatenate
    const totalBytes = buffers.reduce((sum, b) => sum + b.byteLength, 0);
    const data = new ArrayBuffer(totalBytes);
    const out = new Uint8Array(data);
    let offset = 0;
    for (const b of buffers) {
      out.set(new Uint8Array(b), offset);
      offset += b.byteLength;
    }

    return {
      id,
      lodLevel,
      worldPosition: new Float64Array([chunkCenterX, chunkWorldY, chunkCenterZ]),
      data,
    };
  }

  private expandPatchToGPU(
    voxelData: Uint8Array,
    serverLevel: number,
    xOffset: number,
    yOffset: number,
    zOffset: number,
  ): ArrayBuffer {
    const palette = this.metadata!.palette;
    const res = this.resolution;
    const dim = Math.pow(2, serverLevel);
    const step = 64 / dim;
    const voxelSize = res * step;
    const patchWorldSize = 64 * res;

    // Count non-empty
    let count = 0;
    for (let i = 0; i < voxelData.length; i++) {
      if (voxelData[i] !== 0) count++;
    }
    if (count === 0) return new ArrayBuffer(0);

    const data = new ArrayBuffer(count * VOXEL_STRIDE);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);

    let vi = 0;
    for (let i = 0; i < voxelData.length; i++) {
      const val = voxelData[i];
      if (val === 0) continue;

      // Z-fastest iteration: iz = index % dim, iy = floor(index/dim) % dim, ix = floor(index/dim²)
      const iz = i % dim;
      const iy = Math.floor(i / dim) % dim;
      const ix = Math.floor(i / (dim * dim));

      const localX = (ix * step + 0.5 * step) * res - patchWorldSize * 0.5 + xOffset;
      const localY = (iy * step + 0.5 * step) * res - patchWorldSize * 0.5 + yOffset;
      const localZ = (iz * step + 0.5 * step) * res - patchWorldSize * 0.5 + zOffset;

      // Palette lookup (1-indexed)
      const pi = (val - 1) * 3;
      const r = palette[pi];
      const g = palette[pi + 1];
      const b = palette[pi + 2];
      const color = r | (g << 8) | (b << 16) | (0xff << 24);

      const base = vi * 16; // 16 f32s = 64 bytes
      // State A
      f32[base + 0] = localX;
      f32[base + 1] = localY;
      f32[base + 2] = localZ;
      f32[base + 3] = 0;
      u32[base + 4] = color;
      f32[base + 5] = voxelSize;
      f32[base + 6] = 0;
      f32[base + 7] = 0;
      // State B = State A
      f32[base + 8] = localX;
      f32[base + 9] = localY;
      f32[base + 10] = localZ;
      f32[base + 11] = 0;
      u32[base + 12] = color;
      f32[base + 13] = voxelSize;
      f32[base + 14] = 0;
      f32[base + 15] = 0;

      vi++;
    }

    return data;
  }

  private sendRequest(buildMsg: (requestId: number) => ArrayBuffer): Promise<DecodedMessage> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }
      const id = this.nextRequestId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('Request timeout'));
      }, this.requestTimeout);
      this.pending.set(id, { resolve, reject, timer });
      this.ws.send(buildMsg(id));
    });
  }
}
