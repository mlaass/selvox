# selvox — WebGPU Voxel Renderer

## Commands
- `bun install` — install dependencies
- `bun run dev` — start Vite dev server with HMR
- `bun run build` — build library to dist/ (ESM + CJS + types)
- `bun run test` — run tests with vitest
- `bun run typecheck` — type-check without emitting

### Python server (uses uv)
- `cd server && uv run pytest tests/ -v` — run Python tests
- `cd server && uv run voxel-server` — start the voxel server

## Architecture

### Directory layout
- **Library source**: `src/` — published as the `selvox` npm package
- **Demo app**: `demo/` — dev-only, not published
- **Shaders**: `src/shaders/` — WGSL files imported via Vite's `?raw` suffix
- **Entry**: `src/index.ts` re-exports the public API (`VoxelRenderer`, types)

### Rendering pipeline (two-stage GPU)

1. **Compute culling** (`src/shaders/cull.wgsl`, 256-thread workgroups):
   - Frustum-sphere culling (Gribb-Hartmann 6-plane extraction)
   - Projects 8 AABB corners → screen-space billboard bounds
   - Subpixel culling with stochastic probabilistic falloff (configurable threshold)
   - Conditional billboard padding for close-up voxels
   - Outputs: `visible_indices`, `instance_data` (NDC bounds, depth, LOD), `indirect_args` (via atomics)

2. **Render pass** (`src/shaders/voxel.wgsl`):
   - Vertex: reads precomputed instance data, emits 6-vertex billboard quads, reconstructs world-space rays via `inv_view_proj`
   - Fragment: per-pixel ray-AABB intersection (Majercik 2018), slab-based normal determination, directional lighting
   - **Edge supersampling**: at face-junction edges (geometric detection via normalized hit position), casts 4 RGSS sub-pixel rays and averages shaded colors — fast path skips extra work for non-edge pixels
   - Debug modes via bitfield: LOD colors (bit 0), billboard outlines (bit 1), wireframe (bit 2), normals (bit 3), depth (bit 4)
   - `discard` for billboard pixels that miss the AABB; `frag_depth` for correct per-pixel depth

### Key source files
| File | Purpose |
|------|---------|
| `src/VoxelRenderer.ts` | Main renderer: GPU setup, bind groups, uniform management, compute→render orchestration |
| `src/types.ts` | Public interfaces: `RendererOptions`, `IVoxelDataSource`, `VoxelDataChunk` |
| `src/gpu/VoxelPool.ts` | GPU buffer pool for multi-chunk voxel storage, per-LOD stats |
| `src/gpu/BlockAllocator.ts` | First-fit memory allocator with free-list merge (has tests) |
| `src/gpu/context.ts` | WebGPU adapter/device/context initialization |
| `src/gpu/math.ts` | Column-major mat4 utilities (multiply, invert, perspective) |
| `src/ChunkManager.ts` | Spatial LOD streaming: distance-based load/unload with hysteresis |
| `src/FlyCamera.ts` | First-person camera (WASD + mouse look, configurable speed) |
| `src/PerformanceOverlay.ts` | FPS/voxel count HUD (compact/expanded modes) |
| `src/noise.ts` | Seeded Perlin noise with FBM octaves for terrain |
| `src/TerrainSource.ts` | Height-based voxel terrain generator |
| `src/MockOctreeSource.ts` | Procedural octree data source (7 LOD levels) |
| `demo/main.ts` | 16M voxel landscape + houses demo with controls panel |

### Data formats
- **Voxel** (64 bytes): dual-state `pos_a/b`, `color_a/b`, `size_a/b` for interpolation
- **InstanceData** (32 bytes): precomputed NDC bounds, min depth, LOD level
- **Uniforms**: 256-byte aligned — global (view/proj matrices, camera, viewport) + per-chunk (RTE offset, LOD, start slot)
- **Max 512 chunks**, up to 16M voxels in the pool

### Notable design decisions
- **Billboard ray tracing**: no mesh geometry — each voxel is a screen-space quad, fragment shader does ray-AABB
- **RTE (Relative-to-Eye)**: double-precision camera position, per-chunk offsets for large worlds
- **No MSAA**: edge quality handled by shader-only supersampling at cube face junctions
- **Indirect drawing**: compute pass writes `drawIndirect` args via atomics, one draw call per chunk

## Conventions
- Strict TypeScript (`strict: true`, `verbatimModuleSyntax: true`)
- Zero runtime dependencies
- WGSL shaders imported as raw strings: `import shader from './shaders/voxel.wgsl?raw'`
- Library build outputs ESM (`dist/selvox.js`) + CJS (`dist/selvox.cjs`) + types (`dist/index.d.ts`)

## Key References
- PRD: `docs/PRD_WebGPU_Voxel_Renderer.md`
- Paper: `papers/Majercik2018Voxel-lowres.pdf` (ray-box intersection algorithm)
