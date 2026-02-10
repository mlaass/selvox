import type { IVoxelDataSource, RendererOptions } from './types.js';
import { initWebGPU } from './gpu/context.js';

export class VoxelRenderer {
  private canvas: HTMLCanvasElement;
  private options: RendererOptions;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private dataSource: IVoxelDataSource | null = null;

  constructor(canvas: HTMLCanvasElement, options?: RendererOptions) {
    this.canvas = canvas;
    this.options = options ?? {};
  }

  /** Initialize WebGPU device and canvas context */
  async initialize(): Promise<void> {
    const { device, context, format } = await initWebGPU(this.canvas);
    this.device = device;
    this.context = context;
    this.format = format;
  }

  /** Set the data source adapter for streaming voxel data */
  setDataSource(adapter: IVoxelDataSource): void {
    this.dataSource = adapter;
  }

  /** Set the current time for interpolation between keyframes */
  setTime(_timestamp: number): void {
    // TODO: update interpolation_factor uniform
  }

  /** Update camera matrices (e.g. driven by host app / map SDK) */
  updateCamera(
    _viewMatrix: Float32Array,
    _projectionMatrix: Float32Array,
    _positionHigh: Float64Array,
  ): void {
    // TODO: update camera uniforms
  }

  /** Handle canvas resize */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /** Release all GPU resources */
  dispose(): void {
    this.context?.unconfigure();
    this.device?.destroy();
    this.device = null;
    this.context = null;
    this.format = null;
    this.dataSource = null;
  }
}
