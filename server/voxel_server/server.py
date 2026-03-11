from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection

from .protocol import (
    MSG_DELETE_PATCH,
    MSG_HANDSHAKE,
    MSG_LIST_LAYERS,
    MSG_LIST_PATCHES,
    MSG_PUT_PATCH,
    MSG_REQUEST_PATCH,
    MSG_REQUEST_PATCHES,
    encode_batch_patch_data,
    encode_delete_ack,
    encode_error,
    encode_layer_list,
    encode_metadata,
    encode_patch_data,
    encode_patch_list,
    encode_put_ack,
    parse_message,
)
from .voxelgrid import VoxelGrid

log = logging.getLogger(__name__)

MAX_LEVEL = 6

_MSG_NAMES = {
    MSG_HANDSHAKE: "HANDSHAKE",
    MSG_REQUEST_PATCH: "REQUEST_PATCH",
    MSG_REQUEST_PATCHES: "REQUEST_PATCHES",
    MSG_LIST_PATCHES: "LIST_PATCHES",
    MSG_PUT_PATCH: "PUT_PATCH",
    MSG_DELETE_PATCH: "DELETE_PATCH",
    MSG_LIST_LAYERS: "LIST_LAYERS",
}


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
            t0 = time.monotonic()
            detail = ""

            if msg_type == MSG_HANDSHAKE:
                resp = encode_metadata(
                    request_id, grid.world_size, MAX_LEVEL, grid.palette
                )
                await websocket.send(resp)

            elif msg_type == MSG_REQUEST_PATCH:
                layer = msg["layer"]
                detail = f"layer={layer} level={msg['level']} pos=({msg['px']},{msg['py']},{msg['pz']})"
                data = grid.get_patch(layer, msg["px"], msg["py"], msg["pz"], msg["level"])
                resp = encode_patch_data(
                    request_id, layer, msg["level"], msg["px"], msg["py"], msg["pz"], data
                )
                await websocket.send(resp)

            elif msg_type == MSG_REQUEST_PATCHES:
                layer = msg["layer"]
                detail = f"layer={layer} count={len(msg['patches'])}"
                patches = []
                for p in msg["patches"]:
                    data = grid.get_patch(layer, p["px"], p["py"], p["pz"], p["level"])
                    patches.append((p["level"], p["px"], p["py"], p["pz"], data))
                resp = encode_batch_patch_data(request_id, layer, patches)
                await websocket.send(resp)

            elif msg_type == MSG_LIST_PATCHES:
                layer = msg["layer"]
                coords = grid.list_patches(layer)
                detail = f"layer={layer} result_count={len(coords)}"
                resp = encode_patch_list(request_id, layer, coords)
                await websocket.send(resp)

            elif msg_type == MSG_PUT_PATCH:
                layer = msg["layer"]
                px, py, pz = msg["px"], msg["py"], msg["pz"]
                detail = f"layer={layer} pos=({px},{py},{pz})"
                try:
                    raw_data = msg.get("data", b"")
                    voxel_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(64, 64, 64)
                    grid.add_patch(layer, px, py, pz, voxel_data)
                    resp = encode_put_ack(request_id, layer, px, py, pz, 0x00)
                except Exception as e:
                    log.error("PUT_PATCH failed: %s", e)
                    resp = encode_put_ack(request_id, layer, px, py, pz, 0x01)
                await websocket.send(resp)

            elif msg_type == MSG_DELETE_PATCH:
                layer = msg["layer"]
                px, py, pz = msg["px"], msg["py"], msg["pz"]
                detail = f"layer={layer} pos=({px},{py},{pz})"
                grid.delete_patch(layer, px, py, pz)
                resp = encode_delete_ack(request_id, layer, px, py, pz, 0x00)
                await websocket.send(resp)

            elif msg_type == MSG_LIST_LAYERS:
                layers = grid.list_layers()
                resp = encode_layer_list(request_id, layers)
                await websocket.send(resp)

            else:
                await websocket.send(
                    encode_error(request_id, 2, f"Unknown message type: 0x{msg_type:02x}")
                )

            elapsed_ms = (time.monotonic() - t0) * 1000
            log.debug(
                "%s req=%d %s %.1fms",
                _MSG_NAMES.get(msg_type, f"0x{msg_type:02x}"),
                request_id,
                detail,
                elapsed_ms,
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
