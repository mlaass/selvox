export interface WebSocketVoxelSourceOptions {
  url?: string;
  connectTimeout?: number;
  requestTimeout?: number;
  resolution?: number;
  layer?: number;
}

export interface PatchCoord {
  px: number;
  py: number;
  pz: number;
}

export interface ServerMetadata {
  worldSize: [number, number, number];
  maxLevel: number;
  palette: Uint8Array;
}
