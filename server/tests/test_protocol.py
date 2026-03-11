import struct

import numpy as np
import pytest

from voxel_server.protocol import (
    ENC_DENSE,
    ENC_EMPTY,
    MSG_BATCH_PATCH_DATA,
    MSG_ERROR,
    MSG_HANDSHAKE,
    MSG_LIST_PATCHES,
    MSG_METADATA,
    MSG_PATCH_DATA,
    MSG_PATCH_LIST,
    MSG_REQUEST_PATCH,
    MSG_REQUEST_PATCHES,
    encode_batch_patch_data,
    encode_error,
    encode_metadata,
    encode_patch_data,
    encode_patch_list,
    parse_message,
)


class TestParseMessage:
    def test_handshake(self):
        data = struct.pack("<BI", MSG_HANDSHAKE, 42)
        msg = parse_message(data)
        assert msg["type"] == MSG_HANDSHAKE
        assert msg["request_id"] == 42

    def test_request_patch(self):
        data = struct.pack("<BIBiii", MSG_REQUEST_PATCH, 10, 3, 1, 2, 3)
        msg = parse_message(data)
        assert msg["type"] == MSG_REQUEST_PATCH
        assert msg["request_id"] == 10
        assert msg["level"] == 3
        assert msg["px"] == 1
        assert msg["py"] == 2
        assert msg["pz"] == 3

    def test_request_patches(self):
        header = struct.pack("<BIH", MSG_REQUEST_PATCHES, 20, 2)
        p1 = struct.pack("<Biii", 3, 0, 0, 0)
        p2 = struct.pack("<Biii", 5, 1, 2, 3)
        msg = parse_message(header + p1 + p2)
        assert msg["type"] == MSG_REQUEST_PATCHES
        assert len(msg["patches"]) == 2
        assert msg["patches"][0] == {"level": 3, "px": 0, "py": 0, "pz": 0}
        assert msg["patches"][1] == {"level": 5, "px": 1, "py": 2, "pz": 3}

    def test_list_patches(self):
        data = struct.pack("<BIB", MSG_LIST_PATCHES, 99, 4)
        msg = parse_message(data)
        assert msg["type"] == MSG_LIST_PATCHES
        assert msg["level"] == 4

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            parse_message(b"\x01\x00")

    def test_unknown_type(self):
        data = struct.pack("<BI", 0x77, 0)
        with pytest.raises(ValueError, match="Unknown"):
            parse_message(data)


class TestEncodeMetadata:
    def test_basic(self):
        palette = [(255, 0, 0), (0, 255, 0)]
        result = encode_metadata(1, (256, 64, 256), 6, palette)
        assert result[0] == MSG_METADATA
        # Parse it back
        _, req_id, sx, sy, sz, max_level, pal_count = struct.unpack_from("<BI3iBB", result, 0)
        assert req_id == 1
        assert (sx, sy, sz) == (256, 64, 256)
        assert max_level == 6
        assert pal_count == 2
        # Check palette (header: 1+4+12+1+1 = 19 bytes)
        r, g, b = struct.unpack_from("BBB", result, 19)
        assert (r, g, b) == (255, 0, 0)
        r, g, b = struct.unpack_from("BBB", result, 22)
        assert (r, g, b) == (0, 255, 0)


class TestEncodePatchData:
    def test_empty_patch(self):
        result = encode_patch_data(5, 3, 1, 2, 3, None)
        assert result[0] == MSG_PATCH_DATA
        _, req_id, level, px, py, pz = struct.unpack_from("<BIBiii", result, 0)
        assert req_id == 5
        assert level == 3
        assert (px, py, pz) == (1, 2, 3)
        encoding = result[18]
        assert encoding == ENC_EMPTY

    def test_dense_patch(self):
        data = np.full((8, 8, 8), 42, dtype=np.uint8)
        result = encode_patch_data(7, 3, 0, 0, 0, data)
        assert result[0] == MSG_PATCH_DATA
        encoding = result[18]
        assert encoding == ENC_DENSE
        voxel_bytes = result[19:]
        assert len(voxel_bytes) == 512
        assert voxel_bytes[0] == 42


class TestEncodeBatchPatchData:
    def test_batch(self):
        patches = [
            (3, 0, 0, 0, np.full((8, 8, 8), 1, dtype=np.uint8)),
            (3, 1, 0, 0, None),
        ]
        result = encode_batch_patch_data(10, patches)
        assert result[0] == MSG_BATCH_PATCH_DATA
        _, req_id, count = struct.unpack_from("<BIH", result, 0)
        assert req_id == 10
        assert count == 2


class TestEncodePatchList:
    def test_list(self):
        coords = [(0, 0, 0), (1, 2, 3)]
        result = encode_patch_list(15, coords)
        assert result[0] == MSG_PATCH_LIST
        _, req_id, count = struct.unpack_from("<BIH", result, 0)
        assert req_id == 15
        assert count == 2
        px, py, pz = struct.unpack_from("<iii", result, 7)
        assert (px, py, pz) == (0, 0, 0)
        px, py, pz = struct.unpack_from("<iii", result, 19)
        assert (px, py, pz) == (1, 2, 3)


class TestEncodeError:
    def test_error(self):
        result = encode_error(0, 404, "Not found")
        assert result[0] == MSG_ERROR
        _, req_id, code = struct.unpack_from("<BIH", result, 0)
        assert req_id == 0
        assert code == 404
        msg = result[7:].decode("utf-8")
        assert msg == "Not found"
