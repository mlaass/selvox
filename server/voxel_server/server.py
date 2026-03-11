from __future__ import annotations

import asyncio
import logging

import websockets
from websockets.asyncio.server import ServerConnection

from .protocol import (
    MSG_HANDSHAKE,
    MSG_LIST_PATCHES,
    MSG_REQUEST_PATCH,
    MSG_REQUEST_PATCHES,
    encode_batch_patch_data,
    encode_error,
    encode_metadata,
    encode_patch_data,
    encode_patch_list,
    parse_message,
)
from .voxelgrid import VoxelGrid

log = logging.getLogger(__name__)

MAX_LEVEL = 6


async def handle_client(websocket: ServerConnection, grid: VoxelGrid) -> None:
    """Per-client message loop."""
    remote = websocket.remote_address
    log.info("Client connected: %s", remote)
    try:
        async for raw in websocket:
            if isinstance(raw, str):
                continue  # ignore text frames
            try:
                msg = parse_message(raw)
            except ValueError as e:
                await websocket.send(encode_error(0, 1, str(e)))
                continue

            request_id = msg["request_id"]
            msg_type = msg["type"]

            if msg_type == MSG_HANDSHAKE:
                resp = encode_metadata(
                    request_id, grid.world_size, MAX_LEVEL, grid.palette
                )
                await websocket.send(resp)

            elif msg_type == MSG_REQUEST_PATCH:
                data = grid.get_patch(msg["px"], msg["py"], msg["pz"], msg["level"])
                resp = encode_patch_data(
                    request_id, msg["level"], msg["px"], msg["py"], msg["pz"], data
                )
                await websocket.send(resp)

            elif msg_type == MSG_REQUEST_PATCHES:
                patches = []
                for p in msg["patches"]:
                    data = grid.get_patch(p["px"], p["py"], p["pz"], p["level"])
                    patches.append((p["level"], p["px"], p["py"], p["pz"], data))
                resp = encode_batch_patch_data(request_id, patches)
                await websocket.send(resp)

            elif msg_type == MSG_LIST_PATCHES:
                coords = grid.list_patches()
                resp = encode_patch_list(request_id, coords)
                await websocket.send(resp)

            else:
                await websocket.send(
                    encode_error(request_id, 2, f"Unknown message type: 0x{msg_type:02x}")
                )
    except websockets.ConnectionClosed:
        pass
    finally:
        log.info("Client disconnected: %s", remote)


async def start_server(grid: VoxelGrid, host: str, port: int) -> None:
    """Start the WebSocket server."""
    async with websockets.serve(
        lambda ws: handle_client(ws, grid),
        host,
        port,
    ) as server:
        log.info("Listening on ws://%s:%d", host, port)
        await asyncio.Future()  # run forever
