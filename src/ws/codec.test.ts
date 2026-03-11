import { describe, it, expect } from 'vitest';
import {
  encodeHandshake,
  encodeRequestPatch,
  encodeRequestBatch,
  encodeListPatches,
  encodePutPatch,
  encodeDeletePatch,
  encodeListLayers,
  decodeMessage,
  type MetadataMessage,
  type PatchDataMessage,
  type BatchPatchDataMessage,
  type PatchListMessage,
  type PutAckMessage,
  type DeleteAckMessage,
  type LayerListMessage,
  type ErrorMessage,
} from './codec.js';

// Helper: build server messages matching protocol.py's encode functions

function buildMetadata(
  requestId: number,
  worldSize: [number, number, number],
  maxLevel: number,
  palette: [number, number, number][],
): ArrayBuffer {
  // <BI3iBB + palette
  const size = 1 + 4 + 12 + 1 + 1 + palette.length * 3;
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x81);
  dv.setUint32(1, requestId, true);
  dv.setInt32(5, worldSize[0], true);
  dv.setInt32(9, worldSize[1], true);
  dv.setInt32(13, worldSize[2], true);
  dv.setUint8(17, maxLevel);
  dv.setUint8(18, palette.length);
  let offset = 19;
  for (const [r, g, b] of palette) {
    dv.setUint8(offset++, r);
    dv.setUint8(offset++, g);
    dv.setUint8(offset++, b);
  }
  return buf;
}

function buildPatchData(
  requestId: number,
  layer: number,
  level: number,
  px: number,
  py: number,
  pz: number,
  voxelData: Uint8Array | null,
): ArrayBuffer {
  const headerSize = 1 + 4 + 4 + 1 + 12 + 1; // type + reqId + layer + level + coords + encoding
  const dataSize = voxelData ? voxelData.length : 0;
  const buf = new ArrayBuffer(headerSize + dataSize);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x82);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint8(9, level);
  dv.setInt32(10, px, true);
  dv.setInt32(14, py, true);
  dv.setInt32(18, pz, true);
  if (voxelData) {
    dv.setUint8(22, 1); // ENC_DENSE
    new Uint8Array(buf, 23).set(voxelData);
  } else {
    dv.setUint8(22, 0); // ENC_EMPTY
  }
  return buf;
}

function buildBatchPatchData(
  requestId: number,
  layer: number,
  patches: { level: number; px: number; py: number; pz: number; voxelData: Uint8Array | null }[],
): ArrayBuffer {
  let totalSize = 11; // type(1) + reqId(4) + layer(4) + count(2)
  for (const p of patches) {
    totalSize += 14; // per-patch header
    if (p.voxelData) totalSize += p.voxelData.length;
  }
  const buf = new ArrayBuffer(totalSize);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x83);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint16(9, patches.length, true);
  let offset = 11;
  for (const p of patches) {
    dv.setUint8(offset, p.level);
    dv.setInt32(offset + 1, p.px, true);
    dv.setInt32(offset + 5, p.py, true);
    dv.setInt32(offset + 9, p.pz, true);
    if (p.voxelData) {
      dv.setUint8(offset + 13, 1);
      offset += 14;
      new Uint8Array(buf, offset, p.voxelData.length).set(p.voxelData);
      offset += p.voxelData.length;
    } else {
      dv.setUint8(offset + 13, 0);
      offset += 14;
    }
  }
  return buf;
}

function buildPatchList(
  requestId: number,
  layer: number,
  coords: [number, number, number][],
): ArrayBuffer {
  const buf = new ArrayBuffer(11 + coords.length * 12);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x84);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint16(9, coords.length, true);
  let offset = 11;
  for (const [px, py, pz] of coords) {
    dv.setInt32(offset, px, true);
    dv.setInt32(offset + 4, py, true);
    dv.setInt32(offset + 8, pz, true);
    offset += 12;
  }
  return buf;
}

function buildPutAck(
  requestId: number,
  layer: number,
  px: number,
  py: number,
  pz: number,
  status: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(22);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x85);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setInt32(9, px, true);
  dv.setInt32(13, py, true);
  dv.setInt32(17, pz, true);
  dv.setUint8(21, status);
  return buf;
}

function buildDeleteAck(
  requestId: number,
  layer: number,
  px: number,
  py: number,
  pz: number,
  status: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(22);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x86);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setInt32(9, px, true);
  dv.setInt32(13, py, true);
  dv.setInt32(17, pz, true);
  dv.setUint8(21, status);
  return buf;
}

function buildLayerList(
  requestId: number,
  layers: number[],
): ArrayBuffer {
  const buf = new ArrayBuffer(7 + layers.length * 4);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x87);
  dv.setUint32(1, requestId, true);
  dv.setUint16(5, layers.length, true);
  let offset = 7;
  for (const layer of layers) {
    dv.setUint32(offset, layer, true);
    offset += 4;
  }
  return buf;
}

function buildError(requestId: number, code: number, message: string): ArrayBuffer {
  const msgBytes = new TextEncoder().encode(message);
  const buf = new ArrayBuffer(7 + msgBytes.length);
  const dv = new DataView(buf);
  dv.setUint8(0, 0xff);
  dv.setUint32(1, requestId, true);
  dv.setUint16(5, code, true);
  new Uint8Array(buf, 7).set(msgBytes);
  return buf;
}

describe('codec encoders', () => {
  it('encodeHandshake', () => {
    const buf = encodeHandshake(42);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(5);
    expect(dv.getUint8(0)).toBe(0x01);
    expect(dv.getUint32(1, true)).toBe(42);
  });

  it('encodeRequestPatch', () => {
    const buf = encodeRequestPatch(7, 0, 6, -1, 2, 3);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(22);
    expect(dv.getUint8(0)).toBe(0x02);
    expect(dv.getUint32(1, true)).toBe(7);
    expect(dv.getUint32(5, true)).toBe(0); // layer
    expect(dv.getUint8(9)).toBe(6);
    expect(dv.getInt32(10, true)).toBe(-1);
    expect(dv.getInt32(14, true)).toBe(2);
    expect(dv.getInt32(18, true)).toBe(3);
  });

  it('encodeRequestPatch with non-zero layer', () => {
    const buf = encodeRequestPatch(1, 42, 3, 0, 0, 0);
    const dv = new DataView(buf);
    expect(dv.getUint32(5, true)).toBe(42);
    expect(dv.getUint8(9)).toBe(3);
  });

  it('encodeRequestBatch', () => {
    const patches = [
      { level: 6, px: 0, py: 1, pz: 2 },
      { level: 3, px: -5, py: -10, pz: 100 },
    ];
    const buf = encodeRequestBatch(99, 0, patches);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(11 + 13 * 2);
    expect(dv.getUint8(0)).toBe(0x03);
    expect(dv.getUint32(1, true)).toBe(99);
    expect(dv.getUint32(5, true)).toBe(0); // layer
    expect(dv.getUint16(9, true)).toBe(2);
    // First patch
    expect(dv.getUint8(11)).toBe(6);
    expect(dv.getInt32(12, true)).toBe(0);
    expect(dv.getInt32(16, true)).toBe(1);
    expect(dv.getInt32(20, true)).toBe(2);
    // Second patch
    expect(dv.getUint8(24)).toBe(3);
    expect(dv.getInt32(25, true)).toBe(-5);
    expect(dv.getInt32(29, true)).toBe(-10);
    expect(dv.getInt32(33, true)).toBe(100);
  });

  it('encodeListPatches', () => {
    const buf = encodeListPatches(10, 0, 3);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(10);
    expect(dv.getUint8(0)).toBe(0x04);
    expect(dv.getUint32(1, true)).toBe(10);
    expect(dv.getUint32(5, true)).toBe(0); // layer
    expect(dv.getUint8(9)).toBe(3);
  });

  it('encodePutPatch', () => {
    const data = new Uint8Array(8);
    data[0] = 1;
    data[7] = 2;
    const buf = encodePutPatch(50, 5, 1, 2, 3, data);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(22 + 8);
    expect(dv.getUint8(0)).toBe(0x05);
    expect(dv.getUint32(1, true)).toBe(50);
    expect(dv.getUint32(5, true)).toBe(5); // layer
    expect(dv.getInt32(9, true)).toBe(1);
    expect(dv.getInt32(13, true)).toBe(2);
    expect(dv.getInt32(17, true)).toBe(3);
    expect(dv.getUint8(21)).toBe(1); // ENC_DENSE
    expect(dv.getUint8(22)).toBe(1);
    expect(dv.getUint8(29)).toBe(2);
  });

  it('encodeDeletePatch', () => {
    const buf = encodeDeletePatch(60, 3, -1, -2, -3);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(21);
    expect(dv.getUint8(0)).toBe(0x06);
    expect(dv.getUint32(1, true)).toBe(60);
    expect(dv.getUint32(5, true)).toBe(3); // layer
    expect(dv.getInt32(9, true)).toBe(-1);
    expect(dv.getInt32(13, true)).toBe(-2);
    expect(dv.getInt32(17, true)).toBe(-3);
  });

  it('encodeListLayers', () => {
    const buf = encodeListLayers(70);
    const dv = new DataView(buf);
    expect(buf.byteLength).toBe(5);
    expect(dv.getUint8(0)).toBe(0x07);
    expect(dv.getUint32(1, true)).toBe(70);
  });
});

describe('codec decoder', () => {
  it('decodes metadata', () => {
    const buf = buildMetadata(1, [128, -64, 256], 6, [[255, 0, 0], [0, 255, 0]]);
    const msg = decodeMessage(buf) as MetadataMessage;
    expect(msg.type).toBe('metadata');
    expect(msg.requestId).toBe(1);
    expect(msg.worldSize).toEqual([128, -64, 256]);
    expect(msg.maxLevel).toBe(6);
    expect(msg.palette.length).toBe(6);
    expect(Array.from(msg.palette)).toEqual([255, 0, 0, 0, 255, 0]);
  });

  it('decodes metadata with empty palette', () => {
    const buf = buildMetadata(5, [10, 20, 30], 3, []);
    const msg = decodeMessage(buf) as MetadataMessage;
    expect(msg.palette.length).toBe(0);
  });

  it('decodes patch data (dense)', () => {
    const voxels = new Uint8Array([1, 0, 2, 0, 3, 0, 0, 0]); // 2^1 = 2, 2³=8
    const buf = buildPatchData(10, 0, 1, 5, -3, 7, voxels);
    const msg = decodeMessage(buf) as PatchDataMessage;
    expect(msg.type).toBe('patch_data');
    expect(msg.requestId).toBe(10);
    expect(msg.layer).toBe(0);
    expect(msg.level).toBe(1);
    expect(msg.px).toBe(5);
    expect(msg.py).toBe(-3);
    expect(msg.pz).toBe(7);
    expect(msg.encoding).toBe(1);
    expect(msg.voxelData).not.toBeNull();
    expect(Array.from(msg.voxelData!)).toEqual([1, 0, 2, 0, 3, 0, 0, 0]);
  });

  it('decodes patch data with non-zero layer', () => {
    const voxels = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const buf = buildPatchData(11, 99, 1, 0, 0, 0, voxels);
    const msg = decodeMessage(buf) as PatchDataMessage;
    expect(msg.layer).toBe(99);
  });

  it('decodes patch data (empty)', () => {
    const buf = buildPatchData(11, 0, 6, 0, 0, 0, null);
    const msg = decodeMessage(buf) as PatchDataMessage;
    expect(msg.encoding).toBe(0);
    expect(msg.voxelData).toBeNull();
  });

  it('decodes batch patch data', () => {
    const voxels = new Uint8Array(8); // 2^1 cube
    voxels[0] = 1;
    voxels[7] = 2;
    const patches = [
      { level: 1, px: 0, py: 0, pz: 0, voxelData: voxels },
      { level: 1, px: 1, py: 0, pz: 0, voxelData: null },
    ];
    const buf = buildBatchPatchData(20, 0, patches);
    const msg = decodeMessage(buf) as BatchPatchDataMessage;
    expect(msg.type).toBe('batch_patch_data');
    expect(msg.requestId).toBe(20);
    expect(msg.layer).toBe(0);
    expect(msg.patches.length).toBe(2);
    expect(msg.patches[0].encoding).toBe(1);
    expect(msg.patches[0].voxelData![0]).toBe(1);
    expect(msg.patches[0].voxelData![7]).toBe(2);
    expect(msg.patches[1].encoding).toBe(0);
    expect(msg.patches[1].voxelData).toBeNull();
  });

  it('decodes patch list', () => {
    const coords: [number, number, number][] = [[0, 1, 2], [-3, -4, -5]];
    const buf = buildPatchList(30, 0, coords);
    const msg = decodeMessage(buf) as PatchListMessage;
    expect(msg.type).toBe('patch_list');
    expect(msg.requestId).toBe(30);
    expect(msg.layer).toBe(0);
    expect(msg.coords).toEqual([
      { px: 0, py: 1, pz: 2 },
      { px: -3, py: -4, pz: -5 },
    ]);
  });

  it('decodes put ack', () => {
    const buf = buildPutAck(50, 5, 1, 2, 3, 0x00);
    const msg = decodeMessage(buf) as PutAckMessage;
    expect(msg.type).toBe('put_ack');
    expect(msg.requestId).toBe(50);
    expect(msg.layer).toBe(5);
    expect(msg.px).toBe(1);
    expect(msg.py).toBe(2);
    expect(msg.pz).toBe(3);
    expect(msg.status).toBe(0);
  });

  it('decodes delete ack', () => {
    const buf = buildDeleteAck(60, 3, -1, -2, -3, 0x01);
    const msg = decodeMessage(buf) as DeleteAckMessage;
    expect(msg.type).toBe('delete_ack');
    expect(msg.requestId).toBe(60);
    expect(msg.layer).toBe(3);
    expect(msg.px).toBe(-1);
    expect(msg.py).toBe(-2);
    expect(msg.pz).toBe(-3);
    expect(msg.status).toBe(1);
  });

  it('decodes layer list', () => {
    const buf = buildLayerList(70, [0, 5, 42]);
    const msg = decodeMessage(buf) as LayerListMessage;
    expect(msg.type).toBe('layer_list');
    expect(msg.requestId).toBe(70);
    expect(msg.layers).toEqual([0, 5, 42]);
  });

  it('decodes empty layer list', () => {
    const buf = buildLayerList(71, []);
    const msg = decodeMessage(buf) as LayerListMessage;
    expect(msg.layers).toEqual([]);
  });

  it('decodes error', () => {
    const buf = buildError(40, 404, 'Patch not found');
    const msg = decodeMessage(buf) as ErrorMessage;
    expect(msg.type).toBe('error');
    expect(msg.requestId).toBe(40);
    expect(msg.code).toBe(404);
    expect(msg.message).toBe('Patch not found');
  });

  it('handles negative patch coords (i32)', () => {
    const buf = encodeRequestPatch(1, 0, 6, -100, -200, -300);
    const dv = new DataView(buf);
    expect(dv.getInt32(10, true)).toBe(-100);
    expect(dv.getInt32(14, true)).toBe(-200);
    expect(dv.getInt32(18, true)).toBe(-300);
  });

  it('handles max layer value (u32)', () => {
    const buf = encodeRequestPatch(1, 0xFFFFFFFF, 6, 0, 0, 0);
    const dv = new DataView(buf);
    expect(dv.getUint32(5, true)).toBe(0xFFFFFFFF);
  });
});
