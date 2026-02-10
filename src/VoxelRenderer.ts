import type { IVoxelDataSource, RendererOptions } from './types.js';
import { initWebGPU } from './gpu/context.js';
import { mat4Create, mat4Multiply, mat4Invert } from './gpu/math.js';
import { VoxelPool } from './gpu/VoxelPool.js';
import shaderSource from './shaders/voxel.wgsl?raw';

const UNIFORM_BUFFER_SIZE = 256; // padded to 256-byte alignment
const CHUNK_UNIFORM_STRIDE = 256; // 256-byte aligned per chunk
const MAX_CHUNKS = 256;

export class VoxelRenderer {
  private canvas: HTMLCanvasElement;
  private options: RendererOptions;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private dataSource: IVoxelDataSource | null = null;

  // Pipeline state
  private pipeline: GPURenderPipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private chunkBindGroupLayout: GPUBindGroupLayout | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private chunkBindGroup: GPUBindGroup | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private chunkUniformBuffer: GPUBuffer | null = null;
  private pool: VoxelPool | null = null;
  private depthTexture: GPUTexture | null = null;
  private depthTextureView: GPUTextureView | null = null;

  // Camera / uniform data (CPU-side)
  private uniformData = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
  private uniformFloats = new Float32Array(this.uniformData);
  private uniformUints = new Uint32Array(this.uniformData);
  private interpolation = 0;
  private colorInterpolation = 0;
  private debugFlags = 0;

  // RTE state
  private cameraPositionHigh: Float64Array | null = null;

  // CPU-side chunk uniform staging buffer
  private chunkUniformData = new ArrayBuffer(MAX_CHUNKS * CHUNK_UNIFORM_STRIDE);

  constructor(canvas: HTMLCanvasElement, options?: RendererOptions) {
    this.canvas = canvas;
    this.options = options ?? {};
  }

  async initialize(): Promise<void> {
    const { device, context, format } = await initWebGPU(this.canvas);
    this.device = device;
    this.context = context;
    this.format = format;

    const shaderModule = device.createShaderModule({ code: shaderSource });

    // Group 0: global uniforms + voxel storage
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
      ],
    });

    // Group 1: per-chunk uniforms (dynamic offset)
    this.chunkBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout, this.chunkBindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.chunkUniformBuffer = device.createBuffer({
      size: MAX_CHUNKS * CHUNK_UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create pool and bind groups
    this.pool = new VoxelPool(device);

    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.pool.buffer } },
      ],
    });

    this.chunkBindGroup = device.createBindGroup({
      layout: this.chunkBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.chunkUniformBuffer,
            size: CHUNK_UNIFORM_STRIDE,
          },
        },
      ],
    });

    this.createDepthTexture(this.canvas.width, this.canvas.height);
  }

  private createDepthTexture(w: number, h: number): void {
    if (!this.device) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [Math.max(1, w), Math.max(1, h)],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthTextureView = this.depthTexture.createView();
  }

  /** Backward-compatible convenience: loads data as the '__default' chunk. */
  uploadVoxels(data: ArrayBuffer, count: number): void {
    this.loadChunk('__default', data, count);
  }

  loadChunk(
    id: string,
    data: ArrayBuffer,
    count: number,
    worldOrigin?: Float64Array,
    lodLevel?: number,
  ): void {
    if (!this.pool) return;
    this.pool.loadChunk(id, data, count, worldOrigin, lodLevel);
  }

  unloadChunk(id: string): void {
    if (!this.pool) return;
    this.pool.unloadChunk(id);
  }

  get voxelCount(): number {
    return this.pool?.totalVoxels ?? 0;
  }

  get chunkCount(): number {
    return this.pool?.chunkCount ?? 0;
  }

  setTime(t: number): void {
    this.interpolation = t;
  }

  setColorTime(t: number): void {
    this.colorInterpolation = t;
  }

  setDataSource(adapter: IVoxelDataSource): void {
    this.dataSource = adapter;
  }

  setDebugFlags(flags: number): void {
    this.debugFlags = flags;
  }

  get currentDebugFlags(): number {
    return this.debugFlags;
  }

  updateCamera(
    viewMatrix: Float32Array,
    projectionMatrix: Float32Array,
    positionHigh?: Float64Array,
  ): void {
    // If RTE mode, zero out the view translation and store the high-precision position
    let rteView: Float32Array;
    if (positionHigh) {
      this.cameraPositionHigh = positionHigh;
      // Copy view matrix and zero translation (column 3, indices 12-14)
      rteView = new Float32Array(viewMatrix);
      // For a look-at view matrix, the translation is encoded in column 3.
      // We need to recompute: strip the eye-space translation so camera is at origin.
      // view = R * T(-eye). To get just R, we zero the translation part.
      rteView[12] = 0;
      rteView[13] = 0;
      rteView[14] = 0;
    } else {
      this.cameraPositionHigh = null;
      rteView = viewMatrix;
    }

    const viewProj = mat4Create();
    mat4Multiply(viewProj, projectionMatrix, rteView);

    const invViewProj = mat4Create();
    mat4Invert(invViewProj, viewProj);

    const f = this.uniformFloats;
    const u = this.uniformUints;

    // view_proj: offset 0 (16 floats)
    f.set(viewProj, 0);
    // inv_view_proj: offset 16 (16 floats)
    f.set(invViewProj, 16);
    // camera_pos: offset 32 (3 floats)
    if (positionHigh) {
      // RTE: camera at origin
      f[32] = 0;
      f[33] = 0;
      f[34] = 0;
    } else {
      // Extract camera position from inverse view matrix
      const invView = mat4Create();
      mat4Invert(invView, viewMatrix);
      f[32] = invView[12];
      f[33] = invView[13];
      f[34] = invView[14];
    }
    // interpolation: offset 35
    f[35] = this.interpolation;
    // viewport_size: offset 36
    f[36] = this.canvas.width;
    f[37] = this.canvas.height;
    // near: offset 38
    f[38] = 0.1;
    // far: offset 39
    f[39] = 1000.0;
    // color_t: offset 40
    f[40] = this.colorInterpolation;
    // debug_flags: offset 41 (u32)
    u[41] = this.debugFlags;
  }

  render(): void {
    if (
      !this.device || !this.context || !this.pipeline ||
      !this.bindGroup || !this.chunkBindGroup ||
      !this.depthTextureView || !this.pool || !this.chunkUniformBuffer
    ) return;

    // Write global uniforms
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, this.uniformData);

    // Write per-chunk uniforms
    const camHigh = this.cameraPositionHigh;
    this.pool.forEachChunk((_first, _count, chunkInfo) => {
      const byteOffset = chunkInfo.chunkIndex * CHUNK_UNIFORM_STRIDE;
      const view = new Float32Array(this.chunkUniformData, byteOffset, 4);
      if (camHigh) {
        // RTE offset: float32(chunk_origin - camera_pos)
        view[0] = Number(chunkInfo.worldOrigin[0] - camHigh[0]);
        view[1] = Number(chunkInfo.worldOrigin[1] - camHigh[1]);
        view[2] = Number(chunkInfo.worldOrigin[2] - camHigh[2]);
      } else {
        // No RTE: offset is just the world origin
        view[0] = Number(chunkInfo.worldOrigin[0]);
        view[1] = Number(chunkInfo.worldOrigin[1]);
        view[2] = Number(chunkInfo.worldOrigin[2]);
      }
      const u32view = new Uint32Array(this.chunkUniformData, byteOffset + 12, 1);
      u32view[0] = chunkInfo.lodLevel;
    });

    this.device.queue.writeBuffer(this.chunkUniformBuffer, 0, this.chunkUniformData);

    const encoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);

    this.pool.forEachChunk((firstInstance, instanceCount, chunkInfo) => {
      pass.setBindGroup(1, this.chunkBindGroup!, [chunkInfo.chunkIndex * CHUNK_UNIFORM_STRIDE]);
      pass.draw(6, instanceCount, 0, firstInstance);
    });

    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.createDepthTexture(width, height);
  }

  dispose(): void {
    this.depthTexture?.destroy();
    this.pool?.dispose();
    this.uniformBuffer?.destroy();
    this.chunkUniformBuffer?.destroy();
    this.depthTexture = null;
    this.depthTextureView = null;
    this.pool = null;
    this.uniformBuffer = null;
    this.chunkUniformBuffer = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.chunkBindGroup = null;
    this.bindGroupLayout = null;
    this.chunkBindGroupLayout = null;
    this.context?.unconfigure();
    this.device?.destroy();
    this.device = null;
    this.context = null;
    this.format = null;
    this.dataSource = null;
  }
}
