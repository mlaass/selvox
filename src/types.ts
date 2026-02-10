export interface RendererOptions {
  /** Whether to enable anti-aliasing. Default: true */
  antialias?: boolean;
  /** Preferred GPU power profile. Default: 'high-performance' */
  powerPreference?: GPUPowerPreference;
  /** Preferred canvas texture format. If omitted, uses navigator.gpu.getPreferredCanvasFormat() */
  preferredFormat?: GPUTextureFormat;
}

export interface VoxelDataChunk {
  id: string;
  lodLevel: number;
  /** Origin of the chunk in world coordinates (double precision) */
  worldPosition: Float64Array;
  /** Binary data matching the GPU struct layout */
  data: ArrayBuffer;
}

export interface IVoxelDataSource {
  /** Metadata for renderer setup */
  getMetadata(): Promise<{
    worldBounds: Float64Array;
    maxLodDepth: number;
    coordinateSystem: 'cartesian' | 'geospatial';
  }>;

  /** Request data for a specific region, LOD level, and time index */
  requestChunk(
    bbox: Float64Array,
    lodLevel: number,
    timeIndex: number,
  ): Promise<VoxelDataChunk>;
}
