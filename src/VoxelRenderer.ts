import { AAMode, type IVoxelDataSource, type RendererOptions } from './types.js';
import { initWebGPU } from './gpu/context.js';
import { mat4Create, mat4Multiply, mat4Invert } from './gpu/math.js';
import { VoxelPool, type PoolStats } from './gpu/VoxelPool.js';
import shaderSource from './shaders/voxel.wgsl?raw';
import cullSource from './shaders/cull.wgsl?raw';
import blitSource from './shaders/blit.wgsl?raw';
import taaResolveSource from './shaders/taa_resolve.wgsl?raw';
import bilateralSource from './shaders/bilateral.wgsl?raw';
import casSource from './shaders/cas.wgsl?raw';

const UNIFORM_BUFFER_SIZE = 256; // padded to 256-byte alignment
const CHUNK_UNIFORM_STRIDE = 256; // 256-byte aligned per chunk
const MAX_CHUNKS = 512;
const CULL_UNIFORM_SIZE = 256; // padded to 256-byte alignment
const INSTANCE_DATA_STRIDE = 32; // bytes per InstanceData
const INDIRECT_ARGS_STRIDE = 16; // 4 u32s per drawIndirect call
const WORKGROUP_SIZE = 256;
const TAA_UNIFORM_SIZE = 256; // padded to 256-byte alignment
const BILATERAL_UNIFORM_SIZE = 256; // padded to 256-byte alignment
const CAS_UNIFORM_SIZE = 256; // padded to 256-byte alignment
const TAA_HALTON_SEQUENCE_LENGTH = 16;

export class VoxelRenderer {
  private canvas: HTMLCanvasElement;
  private options: RendererOptions;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private dataSource: IVoxelDataSource | null = null;

  // Render pipeline state
  private pipeline: GPURenderPipeline | null = null;
  private pipelineIntermediate: GPURenderPipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private chunkBindGroupLayout: GPUBindGroupLayout | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private chunkBindGroup: GPUBindGroup | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private chunkUniformBuffer: GPUBuffer | null = null;
  private pool: VoxelPool | null = null;
  private depthTexture: GPUTexture | null = null;
  private depthTextureView: GPUTextureView | null = null;

  // MSAA alpha-to-coverage state
  private pipelineMSAA: GPURenderPipeline | null = null;
  private msaaColorTexture: GPUTexture | null = null;
  private msaaColorTextureView: GPUTextureView | null = null;
  private msaaDepthTexture: GPUTexture | null = null;
  private msaaDepthTextureView: GPUTextureView | null = null;

  // Compute culling pipeline state
  private cullPipeline: GPUComputePipeline | null = null;
  private cullBindGroupLayout: GPUBindGroupLayout | null = null;
  private cullBindGroup: GPUBindGroup | null = null;
  private cullUniformBuffer: GPUBuffer | null = null;
  private visibleIndicesBuffer: GPUBuffer | null = null;
  private instanceDataBuffer: GPUBuffer | null = null;
  private indirectArgsBuffer: GPUBuffer | null = null;

  // Staging buffer for clearing indirect args
  private indirectArgsClearData: ArrayBuffer | null = null;

  // AA state
  private aaMode: AAMode = AAMode.None;
  private frameCount = 0;
  private taaHistoryIndex = 0;

  // Post-process textures
  private intermediateColor: GPUTexture | null = null;
  private intermediateColorView: GPUTextureView | null = null;
  private taaHistory: [GPUTexture | null, GPUTexture | null] = [null, null];
  private taaHistoryViews: [GPUTextureView | null, GPUTextureView | null] = [null, null];
  private postProcessOutput: GPUTexture | null = null;
  private postProcessOutputView: GPUTextureView | null = null;

  // Post-process pipelines
  private blitPipeline: GPURenderPipeline | null = null;
  private blitBindGroupLayout: GPUBindGroupLayout | null = null;
  private blitSampler: GPUSampler | null = null;

  private taaPipeline: GPUComputePipeline | null = null;
  private taaBindGroupLayout: GPUBindGroupLayout | null = null;
  private taaUniformBuffer: GPUBuffer | null = null;

  private bilateralPipeline: GPUComputePipeline | null = null;
  private bilateralBindGroupLayout: GPUBindGroupLayout | null = null;
  private bilateralUniformBuffer: GPUBuffer | null = null;

  private casPipeline: GPUComputePipeline | null = null;
  private casBindGroupLayout: GPUBindGroupLayout | null = null;
  private casUniformBuffer: GPUBuffer | null = null;
  private casSharpness = 0.5;

  // TAA jitter state
  private prevViewProjMatrix = new Float32Array(16);
  private hasPrevViewProj = false;

  // Camera / uniform data (CPU-side)
  private uniformData = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
  private uniformFloats = new Float32Array(this.uniformData);
  private uniformUints = new Uint32Array(this.uniformData);
  private interpolation = 0;
  private colorInterpolation = 0;
  private debugFlags = 0;
  private subpixelThreshold = 0.8;
  private voxelScaleFactor = 1.0;
  private bevelRadius = 0.0;

  // Cached view-projection matrix for frustum extraction
  private viewProjMatrix = new Float32Array(16);

  // Draw call tracking
  private lastDrawCallCount = 0;

  // RTE state
  private cameraPositionHigh: Float64Array | null = null;

  // CPU-side chunk uniform staging buffer
  private chunkUniformData = new ArrayBuffer(MAX_CHUNKS * CHUNK_UNIFORM_STRIDE);
  private chunkUniformView = new DataView(this.chunkUniformData);

  // CPU-side cull uniform staging buffer
  private cullUniformData = new ArrayBuffer(MAX_CHUNKS * CULL_UNIFORM_SIZE);
  private cullUniformView = new DataView(this.cullUniformData);

  private maxVoxels: number;

  constructor(canvas: HTMLCanvasElement, options?: RendererOptions) {
    this.canvas = canvas;
    this.options = options ?? {};
    this.maxVoxels = this.options.maxVoxels ?? 2_000_000;
  }

  async initialize(): Promise<void> {
    const { device, context, format } = await initWebGPU(this.canvas);
    this.device = device;
    this.context = context;
    this.format = format;

    const shaderModule = device.createShaderModule({ code: shaderSource });
    const cullModule = device.createShaderModule({ code: cullSource });

    // --- Render pipeline ---

    // Group 0: global uniforms + voxel storage + visible_indices + instance_data
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
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 3,
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
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // Pipeline variant for rendering to rgba16float intermediate (modes 2/3)
    this.pipelineIntermediate = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba16float' }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // Pipeline variant for MSAA 4x alpha-to-coverage (mode 4)
    this.pipelineMSAA = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format! }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      multisample: {
        count: 4,
        alphaToCoverageEnabled: true,
      },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // --- Compute cull pipeline ---

    this.cullBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });

    const cullPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.cullBindGroupLayout],
    });

    this.cullPipeline = device.createComputePipeline({
      layout: cullPipelineLayout,
      compute: {
        module: cullModule,
        entryPoint: 'cull_main',
      },
    });

    // --- Buffers ---

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.chunkUniformBuffer = device.createBuffer({
      size: MAX_CHUNKS * CHUNK_UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.cullUniformBuffer = device.createBuffer({
      size: MAX_CHUNKS * CULL_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.visibleIndicesBuffer = device.createBuffer({
      size: this.maxVoxels * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.instanceDataBuffer = device.createBuffer({
      size: this.maxVoxels * INSTANCE_DATA_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indirectArgsBuffer = device.createBuffer({
      size: MAX_CHUNKS * INDIRECT_ARGS_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // Pre-allocate clear data for indirect args
    this.indirectArgsClearData = new ArrayBuffer(MAX_CHUNKS * INDIRECT_ARGS_STRIDE);

    // Create pool
    this.pool = new VoxelPool(device, this.maxVoxels);

    // Bind groups are created lazily since they depend on the pool buffer
    this.recreateBindGroups();

    // --- Post-process pipelines ---
    this.initPostProcessPipelines(device);

    this.createRenderTargets(this.canvas.width, this.canvas.height);
  }

  private initPostProcessPipelines(device: GPUDevice): void {
    const blitModule = device.createShaderModule({ code: blitSource });
    const taaModule = device.createShaderModule({ code: taaResolveSource });
    const bilateralModule = device.createShaderModule({ code: bilateralSource });

    // --- Blit pipeline ---
    this.blitBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    this.blitPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.blitBindGroupLayout] }),
      vertex: { module: blitModule, entryPoint: 'vs_main' },
      fragment: {
        module: blitModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format! }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    // --- TAA resolve pipeline ---
    this.taaBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    this.taaPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.taaBindGroupLayout] }),
      compute: { module: taaModule, entryPoint: 'taa_main' },
    });

    this.taaUniformBuffer = device.createBuffer({
      size: TAA_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- Bilateral filter pipeline ---
    this.bilateralBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    this.bilateralPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bilateralBindGroupLayout] }),
      compute: { module: bilateralModule, entryPoint: 'bilateral_main' },
    });

    this.bilateralUniformBuffer = device.createBuffer({
      size: BILATERAL_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- CAS sharpening pipeline ---
    const casModule = device.createShaderModule({ code: casSource });

    this.casBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    this.casPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.casBindGroupLayout] }),
      compute: { module: casModule, entryPoint: 'cas_main' },
    });

    this.casUniformBuffer = device.createBuffer({
      size: CAS_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private recreateBindGroups(): void {
    if (
      !this.device || !this.bindGroupLayout || !this.chunkBindGroupLayout ||
      !this.cullBindGroupLayout || !this.pool ||
      !this.uniformBuffer || !this.chunkUniformBuffer || !this.cullUniformBuffer ||
      !this.visibleIndicesBuffer || !this.instanceDataBuffer || !this.indirectArgsBuffer
    ) return;

    // Render bind group (group 0)
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.pool.buffer } },
        { binding: 2, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 3, resource: { buffer: this.instanceDataBuffer } },
      ],
    });

    // Render bind group (group 1) — chunk uniforms with dynamic offset
    this.chunkBindGroup = this.device.createBindGroup({
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

    // Compute cull bind group
    this.cullBindGroup = this.device.createBindGroup({
      layout: this.cullBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.cullUniformBuffer,
            size: CULL_UNIFORM_SIZE,
          },
        },
        { binding: 1, resource: { buffer: this.pool.buffer } },
        { binding: 2, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 3, resource: { buffer: this.instanceDataBuffer } },
        { binding: 4, resource: { buffer: this.indirectArgsBuffer } },
      ],
    });
  }

  private createRenderTargets(w: number, h: number): void {
    if (!this.device) return;
    const width = Math.max(1, w);
    const height = Math.max(1, h);

    // Destroy old textures
    this.depthTexture?.destroy();
    this.intermediateColor?.destroy();
    this.taaHistory[0]?.destroy();
    this.taaHistory[1]?.destroy();
    this.postProcessOutput?.destroy();
    this.msaaColorTexture?.destroy();
    this.msaaDepthTexture?.destroy();

    // Depth texture — depth32float for post-process readability
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthTextureView = this.depthTexture.createView();

    // Intermediate color — render target for modes 2/3
    this.intermediateColor = this.device.createTexture({
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.intermediateColorView = this.intermediateColor.createView();

    // TAA history ping-pong buffers
    for (let i = 0; i < 2; i++) {
      this.taaHistory[i] = this.device.createTexture({
        size: [width, height],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      this.taaHistoryViews[i] = this.taaHistory[i]!.createView();
    }

    // Post-process output (bilateral)
    this.postProcessOutput = this.device.createTexture({
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.postProcessOutputView = this.postProcessOutput.createView();

    // MSAA textures for alpha-to-coverage mode
    if (this.aaMode === AAMode.MSAA_Alpha) {
      this.msaaColorTexture = this.device.createTexture({
        size: [width, height],
        format: this.format!,
        sampleCount: 4,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.msaaColorTextureView = this.msaaColorTexture.createView();

      this.msaaDepthTexture = this.device.createTexture({
        size: [width, height],
        format: 'depth32float',
        sampleCount: 4,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.msaaDepthTextureView = this.msaaDepthTexture.createView();
    } else {
      this.msaaColorTexture = null;
      this.msaaColorTextureView = null;
      this.msaaDepthTexture = null;
      this.msaaDepthTextureView = null;
    }

    // Reset TAA history on resize
    this.hasPrevViewProj = false;
    this.taaHistoryIndex = 0;
  }

  /** Extract 6 frustum planes from a column-major view-projection matrix (Gribb-Hartmann). */
  private extractFrustumPlanes(vp: Float32Array): Float32Array {
    // planes: left, right, bottom, top, near, far — each vec4 (nx, ny, nz, d)
    const planes = new Float32Array(24); // 6 * 4

    // Column-major access: vp[row + col*4]
    // Left:   row3 + row0
    planes[0]  = vp[3]  + vp[0];
    planes[1]  = vp[7]  + vp[4];
    planes[2]  = vp[11] + vp[8];
    planes[3]  = vp[15] + vp[12];

    // Right:  row3 - row0
    planes[4]  = vp[3]  - vp[0];
    planes[5]  = vp[7]  - vp[4];
    planes[6]  = vp[11] - vp[8];
    planes[7]  = vp[15] - vp[12];

    // Bottom: row3 + row1
    planes[8]  = vp[3]  + vp[1];
    planes[9]  = vp[7]  + vp[5];
    planes[10] = vp[11] + vp[9];
    planes[11] = vp[15] + vp[13];

    // Top:    row3 - row1
    planes[12] = vp[3]  - vp[1];
    planes[13] = vp[7]  - vp[5];
    planes[14] = vp[11] - vp[9];
    planes[15] = vp[15] - vp[13];

    // Near (WebGPU [0,1] depth): row2
    planes[16] = vp[2];
    planes[17] = vp[6];
    planes[18] = vp[10];
    planes[19] = vp[14];

    // Far:    row3 - row2
    planes[20] = vp[3]  - vp[2];
    planes[21] = vp[7]  - vp[6];
    planes[22] = vp[11] - vp[10];
    planes[23] = vp[15] - vp[14];

    // Normalize all 6 planes
    for (let i = 0; i < 6; i++) {
      const base = i * 4;
      const len = Math.sqrt(
        planes[base] * planes[base] +
        planes[base + 1] * planes[base + 1] +
        planes[base + 2] * planes[base + 2],
      );
      if (len > 0) {
        const invLen = 1 / len;
        planes[base] *= invLen;
        planes[base + 1] *= invLen;
        planes[base + 2] *= invLen;
        planes[base + 3] *= invLen;
      }
    }

    return planes;
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
    this.recreateBindGroups();
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

  setSubpixelThreshold(value: number): void {
    this.subpixelThreshold = value;
  }

  get currentSubpixelThreshold(): number {
    return this.subpixelThreshold;
  }

  setVoxelScale(factor: number): void {
    this.voxelScaleFactor = Math.max(0.5, Math.min(1.0, factor));
  }

  get currentVoxelScale(): number {
    return this.voxelScaleFactor;
  }

  setBevelRadius(radius: number): void {
    this.bevelRadius = Math.max(0.0, Math.min(0.5, radius));
  }

  get currentBevelRadius(): number {
    return this.bevelRadius;
  }

  setCASSharpness(value: number): void {
    this.casSharpness = Math.max(0.0, Math.min(1.0, value));
  }

  get currentCASSharpness(): number {
    return this.casSharpness;
  }

  get drawCallCount(): number {
    return this.lastDrawCallCount;
  }

  setAAMode(mode: AAMode): void {
    if (this.aaMode !== mode) {
      this.aaMode = mode;
      this.hasPrevViewProj = false;
      this.taaHistoryIndex = 0;
      this.frameCount = 0;
      // Recreate render targets (MSAA textures depend on mode)
      this.createRenderTargets(this.canvas.width, this.canvas.height);
    }
  }

  get currentAAMode(): AAMode {
    return this.aaMode;
  }

  private halton(index: number, base: number): number {
    let result = 0;
    let f = 1 / base;
    let i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  }

  getPoolStats(): PoolStats | null {
    return this.pool?.getStats() ?? null;
  }

  getGpuMemoryStats(): { voxelBuffer: number; visibleIndices: number; instanceData: number; indirectArgs: number; uniforms: number; total: number } | null {
    if (!this.pool) return null;
    const voxelBuffer = this.maxVoxels * 64;
    const visibleIndices = this.maxVoxels * 4;
    const instanceData = this.maxVoxels * INSTANCE_DATA_STRIDE;
    const indirectArgs = MAX_CHUNKS * INDIRECT_ARGS_STRIDE;
    const uniforms = UNIFORM_BUFFER_SIZE + MAX_CHUNKS * CHUNK_UNIFORM_STRIDE + MAX_CHUNKS * CULL_UNIFORM_SIZE;
    return { voxelBuffer, visibleIndices, instanceData, indirectArgs, uniforms, total: voxelBuffer + visibleIndices + instanceData + indirectArgs + uniforms };
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
      rteView = new Float32Array(viewMatrix);
      rteView[12] = 0;
      rteView[13] = 0;
      rteView[14] = 0;
    } else {
      this.cameraPositionHigh = null;
      rteView = viewMatrix;
    }

    // Compute unjittered view-proj first (for TAA prev frame + frustum culling)
    const unjitteredViewProj = mat4Create();
    mat4Multiply(unjitteredViewProj, projectionMatrix, rteView);

    // Save previous unjittered VP for TAA reprojection
    if (this.hasPrevViewProj) {
      this.prevViewProjMatrix.set(this.viewProjMatrix);
    }

    // Cache unjittered VP for frustum extraction (culling always uses unjittered)
    this.viewProjMatrix.set(unjitteredViewProj);

    // Compute jitter for TAA (used in vertex shader position offset, not in VP matrix)
    let jitterX = 0;
    let jitterY = 0;
    if (this.aaMode === AAMode.TAA) {
      const sampleIndex = (this.frameCount % TAA_HALTON_SEQUENCE_LENGTH) + 1;
      jitterX = (this.halton(sampleIndex, 2) - 0.5) * 2.0 / this.canvas.width;
      jitterY = (this.halton(sampleIndex, 3) - 0.5) * 2.0 / this.canvas.height;
    }

    // Always use unjittered VP/inv_VP for main uniforms — stable ray reconstruction
    const unjitteredInvViewProj = mat4Create();
    mat4Invert(unjitteredInvViewProj, unjitteredViewProj);

    const f = this.uniformFloats;
    const u = this.uniformUints;

    // view_proj: offset 0 (16 floats) — always unjittered
    f.set(unjitteredViewProj, 0);
    // inv_view_proj: offset 16 (16 floats) — always unjittered
    f.set(unjitteredInvViewProj, 16);
    // camera_pos: offset 32 (3 floats)
    if (positionHigh) {
      f[32] = 0;
      f[33] = 0;
      f[34] = 0;
    } else {
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
    f[39] = 10000.0;
    // color_t: offset 40
    f[40] = this.colorInterpolation;
    // debug_flags: offset 41 (u32)
    u[41] = this.debugFlags;
    // aa_mode: offset 42 (u32)
    u[42] = this.aaMode;
    // jitter_x: offset 43, jitter_y: offset 44
    f[43] = jitterX;
    f[44] = jitterY;
    // voxel_scale: offset 45
    f[45] = this.voxelScaleFactor;
    // bevel_radius: offset 46
    f[46] = this.bevelRadius;

    // Update TAA uniforms if in TAA mode
    if (this.aaMode === AAMode.TAA && this.taaUniformBuffer && this.device) {
      const taaData = new ArrayBuffer(TAA_UNIFORM_SIZE);
      const taaFloats = new Float32Array(taaData);

      // inv_view_proj (unjittered — matches depth buffer): offset 0
      taaFloats.set(unjitteredInvViewProj, 0);
      // prev_view_proj (unjittered): offset 16
      if (this.hasPrevViewProj) {
        taaFloats.set(this.prevViewProjMatrix, 16);
      } else {
        taaFloats.set(unjitteredViewProj, 16);
      }
      // viewport_size: offset 32
      taaFloats[32] = this.canvas.width;
      taaFloats[33] = this.canvas.height;
      // jitter: offset 34
      taaFloats[34] = jitterX;
      taaFloats[35] = jitterY;
      // blend_factor: offset 36 — 1.0 = first frame (reset), 0.0 = use adaptive blend
      taaFloats[36] = (!this.hasPrevViewProj || this.frameCount === 0) ? 1.0 : 0.0;
      // near: offset 37
      taaFloats[37] = 0.1;
      // far: offset 38
      taaFloats[38] = 10000.0;
      // gamma: offset 39 — variance clipping tightness
      taaFloats[39] = 1.25;

      this.device.queue.writeBuffer(this.taaUniformBuffer, 0, taaData);
    }

    // Update bilateral uniforms if in bilateral mode
    if (this.aaMode === AAMode.Bilateral && this.bilateralUniformBuffer && this.device) {
      const bilData = new ArrayBuffer(BILATERAL_UNIFORM_SIZE);
      const bilFloats = new Float32Array(bilData);
      bilFloats[0] = this.canvas.width;
      bilFloats[1] = this.canvas.height;
      bilFloats[2] = 2.0;  // sigma_spatial
      bilFloats[3] = 0.1;  // sigma_color
      bilFloats[4] = 0.01; // sigma_depth
      this.device.queue.writeBuffer(this.bilateralUniformBuffer, 0, bilData);
    }

    this.hasPrevViewProj = true;
  }

  render(): void {
    if (
      !this.device || !this.context || !this.pipeline || !this.pipelineIntermediate ||
      !this.cullPipeline ||
      !this.bindGroup || !this.chunkBindGroup || !this.cullBindGroup ||
      !this.depthTextureView || !this.pool || !this.chunkUniformBuffer ||
      !this.cullUniformBuffer || !this.indirectArgsBuffer || !this.indirectArgsClearData
    ) return;

    const device = this.device;
    const useMSAA = this.aaMode === AAMode.MSAA_Alpha;
    const usePostProcess = this.aaMode === AAMode.TAA || this.aaMode === AAMode.Bilateral;

    // Write global uniforms
    device.queue.writeBuffer(this.uniformBuffer!, 0, this.uniformData);

    // Extract frustum planes from cached VP matrix
    const frustumPlanes = this.extractFrustumPlanes(this.viewProjMatrix);

    // Write per-chunk uniforms (render) + cull uniforms (compute)
    const camHigh = this.cameraPositionHigh;
    const dv = this.chunkUniformView;
    const cullDv = this.cullUniformView;
    const clearData = new DataView(this.indirectArgsClearData);

    let chunkIndex = 0;
    this.pool.forEachChunk((_first, _count, chunkInfo) => {
      const byteOffset = chunkInfo.chunkIndex * CHUNK_UNIFORM_STRIDE;

      // Compute RTE offset for this chunk
      let rteX: number, rteY: number, rteZ: number;
      if (camHigh) {
        rteX = Number(chunkInfo.worldOrigin[0] - camHigh[0]);
        rteY = Number(chunkInfo.worldOrigin[1] - camHigh[1]);
        rteZ = Number(chunkInfo.worldOrigin[2] - camHigh[2]);
      } else {
        rteX = Number(chunkInfo.worldOrigin[0]);
        rteY = Number(chunkInfo.worldOrigin[1]);
        rteZ = Number(chunkInfo.worldOrigin[2]);
      }

      // Render chunk uniforms
      dv.setFloat32(byteOffset, rteX, true);
      dv.setFloat32(byteOffset + 4, rteY, true);
      dv.setFloat32(byteOffset + 8, rteZ, true);
      dv.setUint32(byteOffset + 12, chunkInfo.lodLevel, true);
      dv.setUint32(byteOffset + 16, chunkInfo.startSlot, true);

      // Cull uniforms for this chunk (256-byte aligned)
      const cullOffset = chunkInfo.chunkIndex * CULL_UNIFORM_SIZE;

      // view_proj: 64 bytes (16 floats)
      for (let i = 0; i < 16; i++) {
        cullDv.setFloat32(cullOffset + i * 4, this.viewProjMatrix[i], true);
      }

      // frustum_planes: 6 * vec4 = 96 bytes at offset 64
      for (let i = 0; i < 24; i++) {
        cullDv.setFloat32(cullOffset + 64 + i * 4, frustumPlanes[i], true);
      }

      // viewport_size: vec2 at offset 160
      cullDv.setFloat32(cullOffset + 160, this.canvas.width, true);
      cullDv.setFloat32(cullOffset + 164, this.canvas.height, true);

      // interpolation: f32 at offset 168
      cullDv.setFloat32(cullOffset + 168, this.interpolation, true);

      // voxel_count: u32 at offset 172
      cullDv.setUint32(cullOffset + 172, chunkInfo.voxelCount, true);

      // rte_offset: vec3 at offset 176
      cullDv.setFloat32(cullOffset + 176, rteX, true);
      cullDv.setFloat32(cullOffset + 180, rteY, true);
      cullDv.setFloat32(cullOffset + 184, rteZ, true);

      // lod_level: u32 at offset 188
      cullDv.setUint32(cullOffset + 188, chunkInfo.lodLevel, true);

      // start_slot: u32 at offset 192
      cullDv.setUint32(cullOffset + 192, chunkInfo.startSlot, true);

      // chunk_index: u32 at offset 196
      cullDv.setUint32(cullOffset + 196, chunkInfo.chunkIndex, true);

      // subpixel_threshold: f32 at offset 200
      cullDv.setFloat32(cullOffset + 200, this.subpixelThreshold, true);

      // aa_mode: u32 at offset 204
      cullDv.setUint32(cullOffset + 204, this.aaMode, true);

      // voxel_scale: f32 at offset 208
      cullDv.setFloat32(cullOffset + 208, this.voxelScaleFactor, true);

      // Clear indirect args for this chunk: vertex_count=6, instance_count=0, first_vertex=0, first_instance=start_slot
      const argsOffset = chunkInfo.chunkIndex * INDIRECT_ARGS_STRIDE;
      clearData.setUint32(argsOffset, 6, true);       // vertex_count
      clearData.setUint32(argsOffset + 4, 0, true);   // instance_count (atomicAdd starts from 0)
      clearData.setUint32(argsOffset + 8, 0, true);   // first_vertex
      clearData.setUint32(argsOffset + 12, 0, true); // first_instance (start_slot now in chunk uniforms)

      chunkIndex++;
    });

    device.queue.writeBuffer(this.chunkUniformBuffer!, 0, this.chunkUniformData);
    device.queue.writeBuffer(this.cullUniformBuffer!, 0, this.cullUniformData);
    device.queue.writeBuffer(this.indirectArgsBuffer, 0, this.indirectArgsClearData);

    const encoder = device.createCommandEncoder();

    // --- Compute pass(es): frustum culling ---
    this.pool.forEachChunk((_first, _count, chunkInfo) => {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.cullPipeline!);
      computePass.setBindGroup(0, this.cullBindGroup!, [chunkInfo.chunkIndex * CULL_UNIFORM_SIZE]);
      const workgroups = Math.ceil(chunkInfo.voxelCount / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroups);
      computePass.end();
    });

    // --- Render pass ---
    let pass: GPURenderPassEncoder;

    if (useMSAA && this.msaaColorTextureView && this.msaaDepthTextureView) {
      pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.msaaColorTextureView,
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: 'clear' as const,
          storeOp: 'discard' as const,
        }],
        depthStencilAttachment: {
          view: this.msaaDepthTextureView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      });
      pass.setPipeline(this.pipelineMSAA!);
    } else {
      const colorTarget = usePostProcess
        ? this.intermediateColorView!
        : this.context.getCurrentTexture().createView();

      pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: colorTarget,
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: 'clear' as const,
          storeOp: 'store' as const,
        }],
        depthStencilAttachment: {
          view: this.depthTextureView!,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: usePostProcess ? 'store' : 'discard',
        },
      });
      pass.setPipeline(usePostProcess ? this.pipelineIntermediate! : this.pipeline!);
    }
    pass.setBindGroup(0, this.bindGroup);

    let drawCalls = 0;
    this.pool.forEachChunk((_first, _count, chunkInfo) => {
      pass.setBindGroup(1, this.chunkBindGroup!, [chunkInfo.chunkIndex * CHUNK_UNIFORM_STRIDE]);
      pass.drawIndirect(this.indirectArgsBuffer!, chunkInfo.chunkIndex * INDIRECT_ARGS_STRIDE);
      drawCalls++;
    });
    this.lastDrawCallCount = drawCalls;

    pass.end();

    // --- Post-process pass (modes 2/3) ---
    if (usePostProcess) {
      let blitSource: GPUTextureView;

      if (this.aaMode === AAMode.TAA) {
        blitSource = this.runTAAResolve(encoder);
      } else {
        blitSource = this.runBilateralFilter(encoder);
      }

      // CAS sharpening after TAA (skip if sharpness is 0)
      if (this.aaMode === AAMode.TAA && this.casSharpness > 0) {
        blitSource = this.runCAS(encoder, blitSource);
      }

      // Blit result to canvas
      this.runBlit(encoder, blitSource);
    }

    device.queue.submit([encoder.finish()]);
    this.frameCount++;
  }

  private runTAAResolve(encoder: GPUCommandEncoder): GPUTextureView {
    const device = this.device!;
    const currentHistoryIdx = this.taaHistoryIndex;
    const prevHistoryIdx = 1 - currentHistoryIdx;
    const outputView = this.taaHistoryViews[currentHistoryIdx]!;

    const bindGroup = device.createBindGroup({
      layout: this.taaBindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.taaUniformBuffer! } },
        { binding: 1, resource: this.intermediateColorView! },
        { binding: 2, resource: this.depthTextureView! },
        { binding: 3, resource: this.taaHistoryViews[prevHistoryIdx]! },
        { binding: 4, resource: this.blitSampler! },
        { binding: 5, resource: outputView },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.taaPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.canvas.width / 8),
      Math.ceil(this.canvas.height / 8),
    );
    pass.end();

    // Flip history index for next frame
    this.taaHistoryIndex = prevHistoryIdx;

    return outputView;
  }

  private runBilateralFilter(encoder: GPUCommandEncoder): GPUTextureView {
    const device = this.device!;

    const bindGroup = device.createBindGroup({
      layout: this.bilateralBindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.bilateralUniformBuffer! } },
        { binding: 1, resource: this.intermediateColorView! },
        { binding: 2, resource: this.depthTextureView! },
        { binding: 3, resource: this.postProcessOutputView! },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.bilateralPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.canvas.width / 8),
      Math.ceil(this.canvas.height / 8),
    );
    pass.end();

    return this.postProcessOutputView!;
  }

  private runCAS(encoder: GPUCommandEncoder, sourceView: GPUTextureView): GPUTextureView {
    const device = this.device!;

    // Write CAS uniforms
    const casData = new ArrayBuffer(CAS_UNIFORM_SIZE);
    const casFloats = new Float32Array(casData);
    casFloats[0] = this.canvas.width;
    casFloats[1] = this.canvas.height;
    casFloats[2] = this.casSharpness;
    device.queue.writeBuffer(this.casUniformBuffer!, 0, casData);

    const bindGroup = device.createBindGroup({
      layout: this.casBindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.casUniformBuffer! } },
        { binding: 1, resource: sourceView },
        { binding: 2, resource: this.postProcessOutputView! },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.casPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.canvas.width / 8),
      Math.ceil(this.canvas.height / 8),
    );
    pass.end();

    return this.postProcessOutputView!;
  }

  private runBlit(encoder: GPUCommandEncoder, sourceView: GPUTextureView): void {
    const bindGroup = this.device!.createBindGroup({
      layout: this.blitBindGroupLayout!,
      entries: [
        { binding: 0, resource: sourceView },
        { binding: 1, resource: this.blitSampler! },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context!.getCurrentTexture().createView(),
        loadOp: 'clear' as const,
        storeOp: 'store' as const,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    pass.setPipeline(this.blitPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.createRenderTargets(width, height);
  }

  dispose(): void {
    this.depthTexture?.destroy();
    this.intermediateColor?.destroy();
    this.taaHistory[0]?.destroy();
    this.taaHistory[1]?.destroy();
    this.postProcessOutput?.destroy();
    this.msaaColorTexture?.destroy();
    this.msaaDepthTexture?.destroy();
    this.pool?.dispose();
    this.uniformBuffer?.destroy();
    this.chunkUniformBuffer?.destroy();
    this.cullUniformBuffer?.destroy();
    this.visibleIndicesBuffer?.destroy();
    this.instanceDataBuffer?.destroy();
    this.indirectArgsBuffer?.destroy();
    this.taaUniformBuffer?.destroy();
    this.bilateralUniformBuffer?.destroy();
    this.casUniformBuffer?.destroy();
    this.depthTexture = null;
    this.depthTextureView = null;
    this.intermediateColor = null;
    this.intermediateColorView = null;
    this.taaHistory = [null, null];
    this.taaHistoryViews = [null, null];
    this.postProcessOutput = null;
    this.postProcessOutputView = null;
    this.pool = null;
    this.uniformBuffer = null;
    this.chunkUniformBuffer = null;
    this.cullUniformBuffer = null;
    this.visibleIndicesBuffer = null;
    this.instanceDataBuffer = null;
    this.indirectArgsBuffer = null;
    this.indirectArgsClearData = null;
    this.taaUniformBuffer = null;
    this.bilateralUniformBuffer = null;
    this.casUniformBuffer = null;
    this.msaaColorTexture = null;
    this.msaaColorTextureView = null;
    this.msaaDepthTexture = null;
    this.msaaDepthTextureView = null;
    this.pipeline = null;
    this.pipelineIntermediate = null;
    this.pipelineMSAA = null;
    this.bindGroup = null;
    this.chunkBindGroup = null;
    this.cullBindGroup = null;
    this.bindGroupLayout = null;
    this.chunkBindGroupLayout = null;
    this.cullBindGroupLayout = null;
    this.cullPipeline = null;
    this.blitPipeline = null;
    this.blitBindGroupLayout = null;
    this.blitSampler = null;
    this.taaPipeline = null;
    this.taaBindGroupLayout = null;
    this.bilateralPipeline = null;
    this.bilateralBindGroupLayout = null;
    this.casPipeline = null;
    this.casBindGroupLayout = null;
    this.context?.unconfigure();
    this.device?.destroy();
    this.device = null;
    this.context = null;
    this.format = null;
    this.dataSource = null;
  }
}
