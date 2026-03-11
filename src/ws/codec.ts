// Binary protocol codec for the voxel WebSocket server.
// All multi-byte values are little-endian.

const ENC_EMPTY = 0;
const ENC_DENSE = 1;

// --- Encoders (Client → Server) ---

export function encodeHandshake(requestId: number): ArrayBuffer {
  const buf = new ArrayBuffer(5);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x01);
  dv.setUint32(1, requestId, true);
  return buf;
}

export function encodeRequestPatch(
  requestId: number,
  layer: number,
  level: number,
  px: number,
  py: number,
  pz: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(22);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x02);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint8(9, level);
  dv.setInt32(10, px, true);
  dv.setInt32(14, py, true);
  dv.setInt32(18, pz, true);
  return buf;
}

export function encodeRequestBatch(
  requestId: number,
  layer: number,
  patches: { level: number; px: number; py: number; pz: number }[],
): ArrayBuffer {
  const buf = new ArrayBuffer(11 + 13 * patches.length);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x03);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint16(9, patches.length, true);
  let offset = 11;
  for (const p of patches) {
    dv.setUint8(offset, p.level);
    dv.setInt32(offset + 1, p.px, true);
    dv.setInt32(offset + 5, p.py, true);
    dv.setInt32(offset + 9, p.pz, true);
    offset += 13;
  }
  return buf;
}

export function encodeListPatches(requestId: number, layer: number, level: number): ArrayBuffer {
  const buf = new ArrayBuffer(10);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x04);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setUint8(9, level);
  return buf;
}

export function encodePutPatch(
  requestId: number,
  layer: number,
  px: number,
  py: number,
  pz: number,
  data: Uint8Array,
): ArrayBuffer {
  const buf = new ArrayBuffer(22 + data.length);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x05);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setInt32(9, px, true);
  dv.setInt32(13, py, true);
  dv.setInt32(17, pz, true);
  dv.setUint8(21, ENC_DENSE);
  new Uint8Array(buf, 22).set(data);
  return buf;
}

export function encodeDeletePatch(
  requestId: number,
  layer: number,
  px: number,
  py: number,
  pz: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(21);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x06);
  dv.setUint32(1, requestId, true);
  dv.setUint32(5, layer, true);
  dv.setInt32(9, px, true);
  dv.setInt32(13, py, true);
  dv.setInt32(17, pz, true);
  return buf;
}

export function encodeListLayers(requestId: number): ArrayBuffer {
  const buf = new ArrayBuffer(5);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x07);
  dv.setUint32(1, requestId, true);
  return buf;
}

// --- Decoder (Server → Client) ---

export interface MetadataMessage {
  type: 'metadata';
  requestId: number;
  worldSize: [number, number, number];
  maxLevel: number;
  palette: Uint8Array;
}

export interface PatchDataMessage {
  type: 'patch_data';
  requestId: number;
  layer: number;
  level: number;
  px: number;
  py: number;
  pz: number;
  encoding: number;
  voxelData: Uint8Array | null;
}

export interface BatchPatchDataMessage {
  type: 'batch_patch_data';
  requestId: number;
  layer: number;
  patches: {
    level: number;
    px: number;
    py: number;
    pz: number;
    encoding: number;
    voxelData: Uint8Array | null;
  }[];
}

export interface PatchListMessage {
  type: 'patch_list';
  requestId: number;
  layer: number;
  coords: { px: number; py: number; pz: number }[];
}

export interface PutAckMessage {
  type: 'put_ack';
  requestId: number;
  layer: number;
  px: number;
  py: number;
  pz: number;
  status: number;
}

export interface DeleteAckMessage {
  type: 'delete_ack';
  requestId: number;
  layer: number;
  px: number;
  py: number;
  pz: number;
  status: number;
}

export interface LayerListMessage {
  type: 'layer_list';
  requestId: number;
  layers: number[];
}

export interface ErrorMessage {
  type: 'error';
  requestId: number;
  code: number;
  message: string;
}

export type DecodedMessage =
  | MetadataMessage
  | PatchDataMessage
  | BatchPatchDataMessage
  | PatchListMessage
  | PutAckMessage
  | DeleteAckMessage
  | LayerListMessage
  | ErrorMessage;

export function decodeMessage(data: ArrayBuffer): DecodedMessage {
  const dv = new DataView(data);
  const msgType = dv.getUint8(0);
  const requestId = dv.getUint32(1, true);

  switch (msgType) {
    case 0x81: {
      const wx = dv.getInt32(5, true);
      const wy = dv.getInt32(9, true);
      const wz = dv.getInt32(13, true);
      const maxLevel = dv.getUint8(17);
      const paletteCount = dv.getUint8(18);
      const palette = new Uint8Array(data, 19, paletteCount * 3);
      return {
        type: 'metadata',
        requestId,
        worldSize: [wx, wy, wz],
        maxLevel,
        palette: new Uint8Array(palette), // copy to detach from buffer
      };
    }
    case 0x82: {
      const layer = dv.getUint32(5, true);
      const level = dv.getUint8(9);
      const px = dv.getInt32(10, true);
      const py = dv.getInt32(14, true);
      const pz = dv.getInt32(18, true);
      const encoding = dv.getUint8(22);
      let voxelData: Uint8Array | null = null;
      if (encoding === ENC_DENSE) {
        voxelData = new Uint8Array(data, 23);
      }
      return { type: 'patch_data', requestId, layer, level, px, py, pz, encoding, voxelData };
    }
    case 0x83: {
      const layer = dv.getUint32(5, true);
      const count = dv.getUint16(9, true);
      const patches: BatchPatchDataMessage['patches'] = [];
      let offset = 11;
      for (let i = 0; i < count; i++) {
        const level = dv.getUint8(offset);
        const px = dv.getInt32(offset + 1, true);
        const py = dv.getInt32(offset + 5, true);
        const pz = dv.getInt32(offset + 9, true);
        const encoding = dv.getUint8(offset + 13);
        offset += 14;
        let voxelData: Uint8Array | null = null;
        if (encoding === ENC_DENSE) {
          const dim = Math.pow(2, level);
          const size = dim * dim * dim;
          voxelData = new Uint8Array(data, offset, size);
          offset += size;
        }
        patches.push({ level, px, py, pz, encoding, voxelData });
      }
      return { type: 'batch_patch_data', requestId, layer, patches };
    }
    case 0x84: {
      const layer = dv.getUint32(5, true);
      const count = dv.getUint16(9, true);
      const coords: PatchListMessage['coords'] = [];
      let offset = 11;
      for (let i = 0; i < count; i++) {
        const px = dv.getInt32(offset, true);
        const py = dv.getInt32(offset + 4, true);
        const pz = dv.getInt32(offset + 8, true);
        coords.push({ px, py, pz });
        offset += 12;
      }
      return { type: 'patch_list', requestId, layer, coords };
    }
    case 0x85: {
      const layer = dv.getUint32(5, true);
      const px = dv.getInt32(9, true);
      const py = dv.getInt32(13, true);
      const pz = dv.getInt32(17, true);
      const status = dv.getUint8(21);
      return { type: 'put_ack', requestId, layer, px, py, pz, status };
    }
    case 0x86: {
      const layer = dv.getUint32(5, true);
      const px = dv.getInt32(9, true);
      const py = dv.getInt32(13, true);
      const pz = dv.getInt32(17, true);
      const status = dv.getUint8(21);
      return { type: 'delete_ack', requestId, layer, px, py, pz, status };
    }
    case 0x87: {
      const count = dv.getUint16(5, true);
      const layers: number[] = [];
      let offset = 7;
      for (let i = 0; i < count; i++) {
        layers.push(dv.getUint32(offset, true));
        offset += 4;
      }
      return { type: 'layer_list', requestId, layers };
    }
    case 0xff: {
      const code = dv.getUint16(5, true);
      const msgBytes = new Uint8Array(data, 7);
      const message = new TextDecoder().decode(msgBytes);
      return { type: 'error', requestId, code, message };
    }
    default:
      throw new Error(`Unknown server message type: 0x${msgType.toString(16)}`);
  }
}
