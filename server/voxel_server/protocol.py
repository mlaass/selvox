"""Binary WebSocket protocol encode/decode per PRD spec.

All messages: [1B type] [4B request_id] [payload...]
Little-endian throughout.
"""

from __future__ import annotations

import struct

import numpy as np

# Client → Server message types
MSG_HANDSHAKE = 0x01
MSG_REQUEST_PATCH = 0x02
MSG_REQUEST_PATCHES = 0x03
MSG_LIST_PATCHES = 0x04

# Server → Client message types
MSG_METADATA = 0x81
MSG_PATCH_DATA = 0x82
MSG_BATCH_PATCH_DATA = 0x83
MSG_PATCH_LIST = 0x84
MSG_ERROR = 0xFF

# Encodings
ENC_EMPTY = 0
ENC_DENSE = 1


def parse_message(data: bytes) -> dict:
    """Parse a client message. Returns dict with 'type', 'request_id', and type-specific fields."""
    if len(data) < 5:
        raise ValueError(f"Message too short: {len(data)} bytes")

    msg_type, request_id = struct.unpack_from("<BI", data, 0)
    result: dict = {"type": msg_type, "request_id": request_id}

    if msg_type == MSG_HANDSHAKE:
        pass  # no payload

    elif msg_type == MSG_REQUEST_PATCH:
        if len(data) < 18:
            raise ValueError("Request Patch message too short")
        level, px, py, pz = struct.unpack_from("<Biii", data, 5)
        result["level"] = level
        result["px"] = px
        result["py"] = py
        result["pz"] = pz

    elif msg_type == MSG_REQUEST_PATCHES:
        if len(data) < 7:
            raise ValueError("Request Patches message too short")
        (count,) = struct.unpack_from("<H", data, 5)
        patches = []
        offset = 7
        for _ in range(count):
            level, px, py, pz = struct.unpack_from("<Biii", data, offset)
            patches.append({"level": level, "px": px, "py": py, "pz": pz})
            offset += 13
        result["patches"] = patches

    elif msg_type == MSG_LIST_PATCHES:
        if len(data) < 6:
            raise ValueError("List Patches message too short")
        (level,) = struct.unpack_from("<B", data, 5)
        result["level"] = level

    else:
        raise ValueError(f"Unknown message type: 0x{msg_type:02x}")

    return result


def encode_metadata(
    request_id: int,
    world_size: tuple[int, int, int],
    max_level: int,
    palette: list[tuple[int, int, int]],
) -> bytes:
    """Encode a Metadata (0x81) response."""
    header = struct.pack(
        "<BI3iBB",
        MSG_METADATA,
        request_id,
        world_size[0],
        world_size[1],
        world_size[2],
        max_level,
        len(palette),
    )
    palette_bytes = b"".join(struct.pack("BBB", r, g, b) for r, g, b in palette)
    return header + palette_bytes


def encode_patch_data(
    request_id: int,
    level: int,
    px: int,
    py: int,
    pz: int,
    voxel_data: np.ndarray | None,
) -> bytes:
    """Encode a Patch Data (0x82) response."""
    header = struct.pack("<BIBiii", MSG_PATCH_DATA, request_id, level, px, py, pz)
    if voxel_data is None:
        return header + struct.pack("<B", ENC_EMPTY)
    else:
        return header + struct.pack("<B", ENC_DENSE) + voxel_data.tobytes(order="C")


def encode_batch_patch_data(
    request_id: int,
    patches: list[tuple[int, int, int, int, np.ndarray | None]],
) -> bytes:
    """Encode a Batch Patch Data (0x83) response.

    patches: list of (level, px, py, pz, voxel_data_or_None)
    """
    header = struct.pack("<BIH", MSG_BATCH_PATCH_DATA, request_id, len(patches))
    parts = [header]
    for level, px, py, pz, voxel_data in patches:
        patch_header = struct.pack("<Biii", level, px, py, pz)
        if voxel_data is None:
            parts.append(patch_header + struct.pack("<B", ENC_EMPTY))
        else:
            parts.append(
                patch_header + struct.pack("<B", ENC_DENSE) + voxel_data.tobytes(order="C")
            )
    return b"".join(parts)


def encode_patch_list(
    request_id: int,
    coords: list[tuple[int, int, int]],
) -> bytes:
    """Encode a Patch List (0x84) response."""
    header = struct.pack("<BIH", MSG_PATCH_LIST, request_id, len(coords))
    coord_bytes = b"".join(struct.pack("<iii", px, py, pz) for px, py, pz in coords)
    return header + coord_bytes


def encode_error(request_id: int, code: int, message: str) -> bytes:
    """Encode an Error (0xFF) response."""
    msg_bytes = message.encode("utf-8")
    return struct.pack("<BIH", MSG_ERROR, request_id, code) + msg_bytes
