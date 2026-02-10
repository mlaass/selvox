import type { IVoxelDataSource, RendererOptions } from './types.js';
import { initWebGPU } from './gpu/context.js';
import { mat4Create, mat4Multiply, mat4Invert } from './gpu/math.js';
import shaderSource from './shaders/voxel.wgsl?raw';

const UNIFORM_BUFFER_SIZE = 256; // padded to 256-byte alignment

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
  private bindGroup: GPUBindGroup | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private voxelStorageBuffer: GPUBuffer | null = null;
  private instanceCount = 0;
  private depthTexture: GPUTexture | null = null;
  private depthTextureView: GPUTextureView | null = null;

  // Camera / uniform data (CPU-side)
  private uniformData = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
  private uniformFloats = new Float32Array(this.uniformData);
  private interpolation = 0;

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

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
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

  uploadVoxels(data: ArrayBuffer, count: number): void {
    if (!this.device || !this.bindGroupLayout || !this.uniformBuffer) return;

    this.voxelStorageBuffer?.destroy();

    this.voxelStorageBuffer = this.device.createBuffer({
      size: Math.max(64, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(this.voxelStorageBuffer.getMappedRange()).set(new Uint8Array(data));
    this.voxelStorageBuffer.unmap();

    this.instanceCount = count;

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.voxelStorageBuffer } },
      ],
    });
  }

  setTime(t: number): void {
    this.interpolation = t;
  }

  setDataSource(adapter: IVoxelDataSource): void {
    this.dataSource = adapter;
  }

  updateCamera(viewMatrix: Float32Array, projectionMatrix: Float32Array): void {
    const viewProj = mat4Create();
    mat4Multiply(viewProj, projectionMatrix, viewMatrix);

    const invViewProj = mat4Create();
    mat4Invert(invViewProj, viewProj);

    // Extract camera position from inverse view matrix
    // inv(view) column 3 = camera world position
    const invView = mat4Create();
    mat4Invert(invView, viewMatrix);

    const f = this.uniformFloats;
    // view_proj: offset 0 (16 floats)
    f.set(viewProj, 0);
    // inv_view_proj: offset 16 (16 floats)
    f.set(invViewProj, 16);
    // camera_pos: offset 32 (3 floats)
    f[32] = invView[12];
    f[33] = invView[13];
    f[34] = invView[14];
    // interpolation: offset 35
    f[35] = this.interpolation;
    // viewport_size: offset 36
    f[36] = this.canvas.width;
    f[37] = this.canvas.height;
    // near: offset 38
    f[38] = 0.1;
    // far: offset 39
    f[39] = 1000.0;
  }

  render(): void {
    if (!this.device || !this.context || !this.pipeline || !this.bindGroup || !this.depthTextureView) return;

    this.device.queue.writeBuffer(this.uniformBuffer!, 0, this.uniformData);

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
    pass.draw(6, this.instanceCount);
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
    this.voxelStorageBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.depthTexture = null;
    this.depthTextureView = null;
    this.voxelStorageBuffer = null;
    this.uniformBuffer = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.bindGroupLayout = null;
    this.instanceCount = 0;
    this.context?.unconfigure();
    this.device?.destroy();
    this.device = null;
    this.context = null;
    this.format = null;
    this.dataSource = null;
  }
}
