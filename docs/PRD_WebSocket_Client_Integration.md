# Product Requirements Document: WebSocket Client Integration

## 1. Summary

A `WebSocketVoxelSource` class that implements `IVoxelDataSource` and connects the selvox renderer to the voxel WebSocket server. Translates server patches (64³ palette-indexed voxel grids) into GPU-ready voxel buffers, integrates with `ChunkManager` for distance-based LOD streaming, and adds a demo UI toggle between demo procedural terrain and WebSocket mode.

All new code lives in `src/` (library) and `demo/` (UI toggle). No server changes required.

## 2. Data Model Alignment

### 2.1 Server Patches → Client Chunks

The server organizes the world as a sparse grid of **64³ patches**, each identified by `(px, py, pz)`. The client's `ChunkManager` manages chunks via `IVoxelDataSource.requestChunk(bbox, lodLevel, timeIndex)`.

Mapping:

| Server concept | Client concept |
|---|---|
| Patch `(px, py, pz)` | Chunk with `id = "ws_L{level}_{px}_{py}_{pz}"` |
| Patch grid coord | `worldPosition = Float64Array([px * 64 * resolution, py * 64 * resolution, pz * 64 * resolution])` |
| `resolution` (world units per voxel) | Voxel `size` field in the 64-byte GPU struct |
| Palette index `0x01`–`0xFF` | Expanded to RGBA via the palette received in the Metadata message |
| Empty voxel `0x00` | Omitted — not written to the GPU buffer |

### 2.2 LOD Mapping

Server LOD levels 0–6 map to resolutions 1³ through 64³. The client's `ChunkManager` requests LOD levels 0–6 where 0 is the finest. The server uses the inverse convention (level 6 = finest, level 0 = coarsest).

Translation: `serverLevel = 6 - clientLevel` (when `maxLodLevel = 6`).

Note: Client LOD 0 = finest resolution. This matches the existing convention used by `TerrainSource` and `MockOctreeSource`.

| Client LOD | Server LOD | Resolution | Voxels per patch | Voxel size (world) |
|---|---|---|---|---|
| 0 | 6 | 64³ | 262,144 | `resolution` |
| 1 | 5 | 32³ | 32,768 | `resolution * 2` |
| 2 | 4 | 16³ | 4,096 | `resolution * 4` |
| 3 | 3 | 8³ | 512 | `resolution * 8` |
| 4 | 2 | 4³ | 64 | `resolution * 16` |
| 5 | 1 | 2³ | 8 | `resolution * 32` |
| 6 | 0 | 1³ | 1 | `resolution * 64` |

### 2.3 3D Grid vs 2D Grid

The current `ChunkManager` iterates a 2D XZ grid (terrain is heightmap-based). Server data is fully 3D. The `WebSocketVoxelSource` must handle the Y axis:

- On handshake, receive world bounds `(size_x, size_y, size_z)` from Metadata
- Compute patch grid extent: `gridY = ceil(size_y / 64)`
- When `requestChunk` is called with a bbox, derive `(px, py, pz)` from the bbox origin
- `ChunkManager` will need minor changes to iterate 3D (or the source can flatten Y into the request — see section 4)

## 3. WebSocketVoxelSource

### 3.1 Class Interface

```typescript
interface WebSocketVoxelSourceOptions {
  /** WebSocket server URL. Default: "ws://localhost:9876" */
  url?: string;
  /** Timeout for initial handshake in ms. Default: 5000 */
  connectTimeout?: number;
}

class WebSocketVoxelSource implements IVoxelDataSource {
  constructor(options?: WebSocketVoxelSourceOptions);

  /** Connect and perform handshake. Must be called before use. */
  connect(): Promise<void>;

  /** Disconnect and clean up. */
  disconnect(): void;

  /** IVoxelDataSource — returns world bounds and LOD info from server metadata. */
  getMetadata(): Promise<{
    worldBounds: Float64Array;
    maxLodDepth: number;
    coordinateSystem: 'cartesian' | 'geospatial';
  }>;

  /** IVoxelDataSource — fetches patch data from server, expands to GPU format. */
  requestChunk(
    bbox: Float64Array,
    lodLevel: number,
    timeIndex: number,
  ): Promise<VoxelDataChunk>;
}
```

### 3.2 Connection Lifecycle

```
construct → connect() → [handshake] → ready
                                         ↕
                               requestChunk() calls
                                         ↓
                                    disconnect()
```

1. `connect()` opens the WebSocket, sends Handshake (`0x01`), waits for Metadata (`0x81`)
2. Stores palette and world size from Metadata response
3. `requestChunk()` sends Request Patch (`0x02`), awaits Patch Data (`0x82`)
4. `disconnect()` closes the socket, rejects any pending requests

### 3.3 Request Multiplexing

Multiple `requestChunk()` calls can be in-flight concurrently. Each sends a unique `request_id` (incrementing u32). A pending-request map resolves the correct Promise when a response arrives:

```typescript
private pending = new Map<number, {
  resolve: (chunk: VoxelDataChunk) => void;
  reject: (err: Error) => void;
}>();
private nextRequestId = 1;
```

Batch requests (`0x03`) should be used when `ChunkManager` issues multiple loads per frame. The source can buffer `requestChunk` calls within a microtask and coalesce them into a single batch message.

### 3.4 Error Handling

- **Connection failure**: `connect()` rejects with an error
- **Server error message** (`0xFF`): reject the corresponding pending request
- **Socket close during operation**: reject all pending requests
- **Timeout**: individual requests time out after 10s (configurable), reject with timeout error

## 4. Binary Protocol Client-Side Codec

### 4.1 Encoder (Client → Server)

```typescript
function encodeHandshake(requestId: number): ArrayBuffer;
function encodeRequestPatch(requestId: number, level: number, px: number, py: number, pz: number): ArrayBuffer;
function encodeRequestBatch(requestId: number, patches: { level: number; px: number; py: number; pz: number }[]): ArrayBuffer;
function encodeListPatches(requestId: number, level: number): ArrayBuffer;
```

All messages: `[1B type][4B request_id][payload...]`, little-endian.

### 4.2 Decoder (Server → Client)

Parse incoming binary frames by reading the first byte (message type):

| Type byte | Handler |
|---|---|
| `0x81` | Parse Metadata → store palette, world size (note: `world_size` fields are **signed i32**, use `getInt32()`) |
| `0x82` | Parse Patch Data → resolve pending request |
| `0x83` | Parse Batch Patch Data → resolve pending requests |
| `0x84` | Parse Patch List → return patch coordinates |
| `0xFF` | Parse Error → reject pending request |

### 4.3 Module Location

```
src/
├── ws/
│   ├── WebSocketVoxelSource.ts   # IVoxelDataSource implementation
│   ├── codec.ts                  # Binary encode/decode functions
│   └── types.ts                  # WS-specific types (ServerMetadata, PatchResponse)
```

Exported from `src/index.ts` as part of the public API.

## 5. Palette Expansion to GPU Format

### 5.1 The Problem

Server sends voxels as single-byte palette indices. The GPU expects 64-byte structs per voxel:

```
Voxel (64 bytes):
  pos_a:   vec4<f32>  (16B) — world position + padding
  color_a: u32        (4B)  — packed RGBA
  size_a:  f32        (4B)  — voxel full extent (diameter); shaders halve internally
  _pad0:   8B
  pos_b:   vec4<f32>  (16B) — same as pos_a (no interpolation)
  color_b: u32        (4B)  — same as color_a
  size_b:  f32        (4B)  — same as size_a
  _pad1:   8B
```

### 5.2 Expansion Algorithm

For each non-empty voxel in the received patch data:

1. **Compute local position** from voxel index within the patch (positions are relative to chunk origin, not world space — the chunk's `worldPosition` provides the RTE offset):
   ```
   dimSize = 2^serverLevel           // e.g. 64 for level 6
   step = 64 / dimSize               // grid cells per voxel at this LOD
   voxelSize = resolution * step
   patchWorldSize = 64 * resolution

   // Server uses C-order (row-major) on shape (X, Y, Z) → Z varies fastest
   iz = index % dimSize
   iy = floor(index / dimSize) % dimSize
   ix = floor(index / dimSize²)

   localX = (ix * step + 0.5 * step) * resolution - patchWorldSize * 0.5
   localY = (iy * step + 0.5 * step) * resolution - patchWorldSize * 0.5
   localZ = (iz * step + 0.5 * step) * resolution - patchWorldSize * 0.5
   ```

2. **Look up palette color**: palette indices in voxel data are 1–255 (0 = empty), so use `palette[paletteIndex - 1]` → `(r, g, b)` → pack as `r | (g << 8) | (b << 16) | (0xFF << 24)`. The `palette_count` byte (0–255) is exactly `len(palette)`; index 0 always means empty and is never in the palette.

3. **Write 64-byte struct**: positions in both A/B slots (no interpolation for static data), color in both slots, size in both slots

4. **Skip empty voxels** (`0x00`): they produce no GPU data

### 5.3 Performance Considerations

- Pre-allocate a reusable `ArrayBuffer` per LOD level (max 262,144 × 64 = 16 MB for level 6)
- Use `DataView` for writing, same pattern as `demo/main.ts`
- The expansion runs on the main thread; for level 6 patches (up to 262K voxels), this takes ~5–10ms on modern hardware
- Future optimization: Web Worker for expansion (not in scope for v1)

### 5.4 RTE (Relative-to-Eye) Integration

The expanded voxel positions are written relative to the patch origin (local space). The chunk's `worldPosition` is set to `(px * 64 * resolution, py * 64 * resolution, pz * 64 * resolution)` as a `Float64Array`, and the renderer applies RTE offsets per-chunk for large-world precision.

## 6. ChunkManager Configuration

### 6.1 Current Design

`ChunkManager` iterates a 2D XZ grid with distance-based LOD rings. Each LOD level doubles the chunk size and radius. This works for heightmap terrain but not for 3D server data.

Currently, `BASE_CHUNK_SIZE = 16` is a hardcoded module-level constant in `ChunkManager.ts`. To support WebSocket mode where patch sizes differ, `ChunkManager` must be refactored to accept `chunkSize` as a constructor parameter via `ChunkManagerOptions`. The module constant becomes the default value, preserving backward compatibility.

### 6.2 Adaptation Strategy

Rather than rewriting ChunkManager for 3D, use a **column-based approach**:

1. `ChunkManager` continues to operate in 2D (XZ grid)
2. For each XZ cell, `WebSocketVoxelSource.requestChunk()` internally requests **all Y-layers** for that column
3. The source fetches all Y-layer patches, expands each to GPU format, and concatenates the buffers into a single merged `VoxelDataChunk`
4. The merged chunk's `worldPosition` uses the XZ column center and the mid-Y of the world bounds (from Metadata `size_y`)
5. The `bbox` Y range from ChunkManager covers the full world height

This keeps ChunkManager changes minimal while supporting 3D data. Since `requestChunk()` returns a single `VoxelDataChunk`, the multi-Y merging is entirely internal to `WebSocketVoxelSource`.

### 6.3 Configuration for Server Data

| Parameter | Demo terrain value | WebSocket value | Rationale |
|---|---|---|---|
| `chunkSize` | 16 (default) | `64 * resolution` | Constructor option; match server patch size |
| `radiusScale` | 48 | `2 * patchWorldSize` | See worked example in section 7.1 |
| `maxChunksPerFrame` | 4 | 2 | Network latency limits throughput |
| `maxLodLevel` | 6 | 6 | Match server's 7 LOD levels |

### 6.4 Patch Discovery

On connect, the source should request the patch list (`0x04`) at a coarse LOD level to know which patches exist. This avoids requesting empty patches:

```typescript
private knownPatches: Set<string>;  // "px_py_pz"

async connect(): Promise<void> {
  // ... handshake ...
  // Request patch list at coarsest level to discover occupied patches
  this.knownPatches = await this.requestPatchList(0);
}
```

`requestChunk()` checks `knownPatches` before sending a server request. If no patches exist at the requested XZ column, return an empty chunk immediately.

## 7. Distance-Based LOD Configuration

### 7.1 LOD Ring Distances

The `ChunkManager` uses `radiusScale * (1 << level)` for each LOD ring's outer radius. With server data, the distances should be tuned for patch size.

#### Worked Example: `resolution = 0.5`

```
patchWorldSize = 64 * 0.5 = 32 world units
chunkSize = patchWorldSize = 32  (set via ChunkManagerOptions)
radiusScale = 2 * patchWorldSize = 64
```

**Ring distance table:**

| Client LOD | Server LOD | Ring outer radius | Ring distance (patches) | Voxel size |
|---|---|---|---|---|
| 0 (finest) | 6 (64³) | `64 * 1 = 64` | 2 patches | 0.5 |
| 1 | 5 (32³) | `64 * 2 = 128` | 4 patches | 1.0 |
| 2 | 4 (16³) | `64 * 4 = 256` | 8 patches | 2.0 |
| 3 | 3 (8³) | `64 * 8 = 512` | 16 patches | 4.0 |
| 4 | 2 (4³) | `64 * 16 = 1024` | 32 patches | 8.0 |
| 5 | 1 (2³) | `64 * 32 = 2048` | 64 patches | 16.0 |
| 6 (coarsest) | 0 (1³) | `64 * 64 = 4096` | 128 patches | 32.0 |

**Grid cell / patch size alignment:** When `chunkSize` is set to `patchWorldSize` (e.g., 32), ChunkManager grid cells align 1:1 with server patches at LOD 0. At coarser LOD levels, ChunkManager doubles the cell size per level (`chunkSize * (1 << level)`), which naturally matches the server's coarser patches covering the same world extent. If `chunkSize` does not match `patchWorldSize`, the source must map each ChunkManager cell to the overlapping server patch(es) — this is avoided by setting `chunkSize = 64 * resolution`.

### 7.2 Adaptive Loading

For large datasets, not all LOD 0 patches will fit in the GPU pool (16M voxel limit). The source should:

1. Start by loading all patches at a coarse LOD (level 3 = 8³ = 512 voxels per patch)
2. Progressively refine nearby patches to finer LODs as the camera moves
3. Unload fine LOD data when the camera moves away (ChunkManager handles this via hysteresis)

### 7.3 Budget Awareness

The renderer's default voxel pool is **16M voxels**. Budget estimates:

| LOD level | Max voxels per patch | Typical non-empty % | Effective voxels | Patches @ 16M pool |
|---|---|---|---|---|
| 0 (64³) | 262,144 | 10–30% | ~26K–79K | 200–615 |
| 1 (32³) | 32,768 | 20–50% | ~6K–16K | 1000–2666 |
| 2 (16³) | 4,096 | 30–70% | ~1K–3K | 5333–16000 |
| 3 (8³) | 512 | 50–90% | ~256–460 | 34783–62500 |

## 8. Demo UI Toggle

### 8.1 Mode Selection

Add a dropdown or toggle to `demo/main.ts` that switches between:

- **Demo** (default): current procedural terrain + houses generation
- **WebSocket**: connects to the voxel server

### 8.2 UI Design

A simple `<select>` in the existing controls panel (toggled with `O`):

```
┌─ Controls ─────────────┐
│ Source [Demo ▾]         │
│ Server [ws://...:9876]  │  ← only visible in WebSocket mode
│ [Connect]               │  ← only visible in WebSocket mode
│ ─────────────────────── │
│ LOD colors  ☐           │
│ ... existing controls ...│
└─────────────────────────┘
```

### 8.3 Mode Switch Behavior

1. **Demo → WebSocket**:
   - Unload all current chunks (`renderer.unloadAll()` or per-chunk unload)
   - Create `WebSocketVoxelSource`, call `connect()`
   - Create new `ChunkManager` with the WebSocket source and server-tuned options
   - Begin streaming loop

2. **WebSocket → Demo**:
   - `ChunkManager.unloadAll()`
   - `WebSocketVoxelSource.disconnect()`
   - Regenerate and load demo terrain chunks

3. **Connection failure**: show error in the controls panel, fall back to Demo mode

### 8.4 URL Configuration

The server URL defaults to `ws://localhost:9876` and is editable via a text input in the controls panel. Stored in `localStorage` for persistence across page reloads.

## 9. File Changes

### New Files

| File | Purpose |
|---|---|
| `src/ws/WebSocketVoxelSource.ts` | `IVoxelDataSource` implementation over WebSocket |
| `src/ws/codec.ts` | Binary protocol encode/decode |
| `src/ws/types.ts` | WebSocket-specific type definitions |

### Modified Files

| File | Change |
|---|---|
| `src/index.ts` | Re-export `WebSocketVoxelSource` and WS types |
| `src/ChunkManager.ts` | Add `chunkSize` to `ChunkManagerOptions`; use it instead of hardcoded `BASE_CHUNK_SIZE` |
| `demo/main.ts` | Add source toggle UI, WebSocket mode, ChunkManager integration |

### Unchanged

- `src/VoxelRenderer.ts` — no changes needed, already supports `loadChunk`/`unloadChunk`
- Server code — no modifications required

## 10. Development Plan

### Phase 1: Protocol Codec
- `src/ws/codec.ts`: encode/decode all message types
- Unit tests for round-trip encoding

### Phase 2: WebSocketVoxelSource
- `src/ws/WebSocketVoxelSource.ts`: connection, handshake, patch requests
- `src/ws/types.ts`: type definitions
- Palette expansion logic (palette index → 64-byte GPU struct)
- Integration test against running server

### Phase 3: Demo Integration
- Source toggle UI in `demo/main.ts`
- ChunkManager wiring for WebSocket mode
- End-to-end test: server with LAS data → client renders voxels

### Phase 4: Polish
- Batch request coalescing (microtask buffering)
- Connection status indicator in the UI
- Error recovery (auto-reconnect on disconnect)
