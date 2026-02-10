# selvox

High-performance WebGPU geospatial voxel visualization library.

## Overview

selvox renders large-scale, time-series volumetric data (weather simulations, urban growth, sensor measurements) using the **Majercik 2018 ray-box intersection** algorithm. Unlike traditional voxel engines, it uses no meshing — voxels are ray-traced bounding boxes with GPU-driven interpolation between keyframes.

### Key Architecture Decisions

- **No meshing** — voxels rendered as ray-cast oriented bounding boxes
- **GPU interpolation** — smooth transitions between time steps without re-uploading geometry
- **Relative-To-Eye (RTE)** coordinates — handles geospatial-scale coordinates without float jitter
- **Framework-agnostic** — embeddable in any web app (React, Vue, Svelte, vanilla JS)
- **Zero runtime dependencies**

## Prerequisites

- [Bun](https://bun.sh/) (runtime & package manager)
- A WebGPU-capable browser (Chrome 113+, Edge 113+, or Firefox Nightly with flags)

## Quick Start

```bash
bun install
bun run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`).

## Project Structure

```
src/
├── index.ts              # Library entry — re-exports public API
├── VoxelRenderer.ts      # Core public API class
├── types.ts              # Shared interfaces
├── gpu/
│   └── context.ts        # WebGPU initialization
└── shaders/
    ├── voxel.wgsl        # WGSL shaders
    └── wgsl.d.ts         # Type declarations for ?raw imports
demo/
└── main.ts               # Demo app (dev only)
```

## Scripts

| Command | Description |
|---------|-------------|
| `bun run dev` | Start dev server with HMR |
| `bun run build` | Build library (ESM + CJS + types) |
| `bun run preview` | Preview production build |
| `bun run test` | Run tests |
| `bun run typecheck` | Type-check without emitting |

## Development Roadmap

1. **Phase 1** — Interpolated Cube: single cube with smooth size/color transitions
2. **Phase 2** — Adapter & Allocator: load variable-sized chunks via `IVoxelDataSource`
3. **Phase 3** — Geospatial Context: octree backend, RTE coordinates, LOD switching
4. **Phase 4** — Integration: hybrid rendering with terrain/map base layers

## License

MIT
