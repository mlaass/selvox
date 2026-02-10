# selvox — WebGPU Voxel Renderer

## Commands
- `bun install` — install dependencies
- `bun run dev` — start Vite dev server with HMR
- `bun run build` — build library to dist/ (ESM + CJS + types)
- `bun run test` — run tests with vitest
- `bun run typecheck` — type-check without emitting

## Architecture
- **Library source**: `src/` — published as the `selvox` npm package
- **Demo app**: `demo/` — dev-only, not published
- **Shaders**: `src/shaders/` — WGSL files imported via Vite's `?raw` suffix
- **Entry**: `src/index.ts` re-exports the public API (`VoxelRenderer`, types)

## Conventions
- Strict TypeScript (`strict: true`, `verbatimModuleSyntax: true`)
- Zero runtime dependencies
- WGSL shaders imported as raw strings: `import shader from './shaders/voxel.wgsl?raw'`
- Library build outputs ESM (`dist/selvox.js`) + CJS (`dist/selvox.cjs`) + types (`dist/index.d.ts`)

## Key References
- PRD: `docs/PRD_WebGPU_Voxel_Renderer.md`
- Paper: `papers/Majercik2018Voxel-lowres.pdf` (ray-box intersection algorithm)
