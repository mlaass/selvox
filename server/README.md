# voxel-server

WebSocket server that streams voxel data to the selvox frontend. It can serve either a LAS/LAZ point cloud file or procedurally generated terrain.

## Install

```sh
cd server
uv sync
```

For development (pytest):

```sh
uv sync --extra dev
```

## Run

**Procedural terrain** (no data file needed):

```sh
uv run python -m voxel_server
```

**LAS/LAZ point cloud**:

```sh
uv run python -m voxel_server --file data/tile.laz
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | _(none)_ | LAS/LAZ file to load |
| `--resolution` | `1.0` | World units per voxel |
| `--port` | `9876` | WebSocket port |
| `--terrain-size` | `256` | Procedural terrain grid size (when no file given) |

## Download test dataset

A script is included to fetch a small (~23 MB) USGS 3DEP LAZ tile:

```sh
uv run python scripts/download_dataset.py
```

This saves to `data/tile.laz`.

## Connecting from the frontend

Start the dev server (`bun run dev` in the repo root), then point the frontend's WebSocket client at `ws://localhost:9876`.

## Tests

```sh
uv run pytest
```
