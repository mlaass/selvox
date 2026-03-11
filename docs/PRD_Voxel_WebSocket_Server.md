# Product Requirements Document: Voxel Octree WebSocket Server

## 1. Summary

A minimal Python WebSocket server that loads a LAS/LAZ point cloud, voxelizes it, stores the result as 64³ patches in a flat dict, and serves patches to clients at multiple LOD levels. Proof of concept — kept as simple as possible.

The server lives in `server/` as a standalone Python project managed with **uv**.

## 2. Data Model

### 2.1 Patch Grid

The voxelized world is divided into a regular grid of **64³ patches**. Each patch is identified by its integer grid coordinate `(px, py, pz)`. Coordinates are simple indices: `(0,0,0)`, `(1,0,0)`, `(2,0,0)`, etc. — not world positions. The client derives world position as `patch_coord * 64 * resolution`.

Storage is a flat Python dict — no tree structure, no pointers, no traversal:

```python
# Only non-empty patches exist in the dict (sparse at patch level)
patches: dict[tuple[int, int, int], np.ndarray]  # key → uint8 shape (64,64,64)

# Example: a 256×64×256 voxel world = 4×1×4 grid = up to 16 patches
patches[(0, 0, 0)] = np.array(...)  # 64³ uint8
patches[(1, 0, 0)] = np.array(...)
patches[(3, 0, 3)] = np.array(...)
# missing keys → empty space, never stored
```

### 2.2 Voxel Representation

Each voxel is a **single byte**:
- `0x00` = empty
- `0x01`–`0xFF` = palette color index

### 2.3 LOD Levels (precomputed)

Each patch has 7 LOD levels, all **precomputed at load time** and stored alongside the full-resolution data:

| Level | Resolution | Voxels | Bytes |
|-------|-----------|--------|-------|
| 6     | 64³       | 262,144 | 256 KB |
| 5     | 32³       | 32,768 | 32 KB |
| 4     | 16³       | 4,096 | 4 KB |
| 3     | 8³        | 512 | 512 B |
| 2     | 4³        | 64 | 64 B |
| 1     | 2³        | 8 | 8 B |
| 0     | 1³        | 1 | 1 B |

Total per patch: ~293 KB (only 14% overhead vs storing level 6 alone).

**Downsampling**: each coarser level is built from the level above by taking the most common non-empty value in each 2³ block (majority vote). Computed once during ingestion, stored as a list of 7 arrays per patch:

```python
# Per-patch LOD pyramid, index 0 = coarsest (1³), index 6 = finest (64³)
patch_lods: dict[tuple[int, int, int], list[np.ndarray]]

# Serving a request is just a dict lookup + index:
def get_patch(px, py, pz, level):
    lods = patch_lods.get((px, py, pz))
    if lods is None:
        return None  # empty patch
    return lods[level]  # already precomputed, just send it
```

### 2.4 Palette

A global palette of up to 255 RGB colors. Generated automatically during ingestion by quantizing point cloud colors (or height-based coloring if no color attribute).

## 3. Data Ingestion

### 3.1 LAS/LAZ Loader

Single supported format. Uses `laspy` to read point clouds.

**Pipeline:**
```
LAS/LAZ file → laspy reader → point positions + colors
  → quantize positions to voxel grid (configurable resolution)
  → quantize colors to 255-entry palette (k-means or uniform)
  → fill 64³ patches → store in dict
```

The voxel resolution (world units per voxel) is configurable. Points mapping to the same voxel cell are merged (last-write-wins or majority color).

### 3.2 Demo Dataset

A download script (`server/scripts/download_dataset.py`) fetches a freely available LAS/LAZ tile from **USGS 3DEP** (US geological survey LiDAR). These are public domain, typically 1–50M points per tile, and cover real terrain — ideal for a GIS proof of concept.

Fallback: a simple procedural terrain generator using numpy (no extra dependencies) for testing without downloading anything.

## 4. WebSocket Protocol

### 4.1 Transport

- **Library**: `websockets`
- **Port**: 9876 (configurable)
- **Messages**: binary frames, little-endian

### 4.2 Message Format

All messages: `[1B type] [4B request_id] [payload...]`

### 4.3 Client → Server

| Type | Name | Payload |
|------|------|---------|
| `0x01` | Handshake | *(none)* |
| `0x02` | Request Patch | `[1B level] [4B px] [4B py] [4B pz]` |
| `0x03` | Request Patches (batch) | `[2B count] [count × (1B level, 4B px, 4B py, 4B pz)]` |
| `0x04` | List Patches | `[1B level]` — list all non-empty patches |

### 4.4 Server → Client

| Type | Name | Payload |
|------|------|---------|
| `0x81` | Metadata | `[4B size_x] [4B size_y] [4B size_z] [1B max_level] [1B palette_count] [N × 3B RGB]` |
| `0x82` | Patch Data | `[1B level] [4B px] [4B py] [4B pz] [1B encoding] [data]` |
| `0x83` | Batch Patch Data | `[2B count] [count × patch_data]` |
| `0x84` | Patch List | `[2B count] [count × (4B px, 4B py, 4B pz)]` |
| `0xFF` | Error | `[2B code] [UTF-8 message]` |

**Encodings** for patch data:
- `0` = empty (no data)
- `1` = dense (raw `(2^level)³` bytes, x-fastest order)

### 4.5 Typical Flow

```
Client                          Server
  |--- Handshake (0x01) --------->|
  |<-- Metadata (0x81) -----------|  (world size, palette)
  |--- List Patches (0x04) ----->|  (level=3)
  |<-- Patch List (0x84) --------|  (which patches exist)
  |--- Request Patches (0x03) -->|  (batch of patches at level 3)
  |<-- Batch Patch Data (0x83) --|  (voxel data)
```

## 5. Server Architecture

### 5.1 Project Structure

```
server/
├── pyproject.toml
├── scripts/
│   └── download_dataset.py
├── voxel_server/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── server.py            # WebSocket handler
│   ├── protocol.py          # Binary message encode/decode
│   ├── voxelgrid.py         # Patch dict + precomputed LOD pyramid
│   ├── ingest.py            # LAS/LAZ → patches
│   └── terrain.py           # Procedural fallback generator
└── tests/
    ├── test_voxelgrid.py
    └── test_protocol.py
```

### 5.2 Dependencies

```toml
[project]
name = "voxel-server"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "websockets>=14.0",
    "numpy>=2.0",
    "laspy[lazrs]>=2.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[project.scripts]
voxel-server = "voxel_server.__main__:main"
```

Three runtime dependencies: `websockets`, `numpy`, `laspy` (with lazrs backend for .laz decompression).

### 5.3 Implementation Notes

- Fully async (`asyncio` + `websockets`)
- Patches dict is built at startup, then immutable — no locking needed
- CLI: `uv run voxel-server --file scan.laz --resolution 0.5 --port 9876`
- `--resolution`: world units per voxel (default: 1.0)
- All 7 LOD levels precomputed at startup; serving a request is just `dict[coords][level]` → send bytes

## 6. Client Integration (informational, not in scope)

The selvox `IVoxelDataSource` interface can be adapted to connect to this server. A future `WebSocketVoxelSource` would translate `requestChunk()` calls into patch requests and expand palette indices to RGBA for the GPU struct.

## 7. Development Plan

### Phase 1: Core
- Set up `server/` with uv
- `voxelgrid.py`: patch dict + LOD downsampling
- `protocol.py`: binary message encode/decode
- `server.py`: WebSocket server with all message types
- `terrain.py`: procedural fallback for testing
- Tests

### Phase 2: LAS Ingestion
- `ingest.py`: LAS/LAZ loading + voxelization + palette generation
- `download_dataset.py`: fetch a USGS 3DEP tile
- End-to-end test with real data
