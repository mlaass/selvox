import struct

import numpy as np
import pytest

from voxel_server.protocol import (
    ENC_DENSE,
    ENC_EMPTY,
    MSG_BATCH_PATCH_DATA,
    MSG_DELETE_ACK,
    MSG_DELETE_PATCH,
    MSG_ERROR,
    MSG_HANDSHAKE,
    MSG_LAYER_LIST,
    MSG_LIST_LAYERS,
    MSG_LIST_PATCHES,
    MSG_METADATA,
    MSG_PATCH_DATA,
    MSG_PATCH_LIST,
    MSG_PUT_ACK,
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


class TestParseMessage:
    def test_handshake(self):
        data = struct.pack("<BI", MSG_HANDSHAKE, 42)
        msg = parse_message(data)
        assert msg["type"] == MSG_HANDSHAKE
        assert msg["request_id"] == 42

    def test_request_patch(self):
        data = struct.pack("<BIIBiii", MSG_REQUEST_PATCH, 10, 0, 3, 1, 2, 3)
        msg = parse_message(data)
        assert msg["type"] == MSG_REQUEST_PATCH
        assert msg["request_id"] == 10
        assert msg["layer"] == 0
        assert msg["level"] == 3
        assert msg["px"] == 1
        assert msg["py"] == 2
        assert msg["pz"] == 3

    def test_request_patches(self):
        header = struct.pack("<BIIH", MSG_REQUEST_PATCHES, 20, 0, 2)
        p1 = struct.pack("<Biii", 3, 0, 0, 0)
        p2 = struct.pack("<Biii", 5, 1, 2, 3)
        msg = parse_message(header + p1 + p2)
        assert msg["type"] == MSG_REQUEST_PATCHES
        assert msg["layer"] == 0
        assert len(msg["patches"]) == 2
        assert msg["patches"][0] == {"level": 3, "px": 0, "py": 0, "pz": 0}
        assert msg["patches"][1] == {"level": 5, "px": 1, "py": 2, "pz": 3}

    def test_list_patches(self):
        data = struct.pack("<BIIB", MSG_LIST_PATCHES, 99, 0, 4)
        msg = parse_message(data)
        assert msg["type"] == MSG_LIST_PATCHES
        assert msg["layer"] == 0
        assert msg["level"] == 4

    def test_put_patch_empty(self):
        data = struct.pack("<BIIiiiB", MSG_PUT_PATCH, 30, 0, 1, 2, 3, ENC_EMPTY)
        msg = parse_message(data)
        assert msg["type"] == MSG_PUT_PATCH
        assert msg["request_id"] == 30
        assert msg["layer"] == 0
        assert (msg["px"], msg["py"], msg["pz"]) == (1, 2, 3)
        assert msg["encoding"] == ENC_EMPTY
        assert "data" not in msg

    def test_put_patch_dense(self):
        voxels = bytes(range(256)) * 2  # 512 bytes
        header = struct.pack("<BIIiiiB", MSG_PUT_PATCH, 31, 2, 0, 0, 0, ENC_DENSE)
        msg = parse_message(header + voxels)
        assert msg["type"] == MSG_PUT_PATCH
        assert msg["layer"] == 2
        assert msg["encoding"] == ENC_DENSE
        assert msg["data"] == voxels

    def test_delete_patch(self):
        data = struct.pack("<BIIiii", MSG_DELETE_PATCH, 40, 1, 4, 5, 6)
        msg = parse_message(data)
        assert msg["type"] == MSG_DELETE_PATCH
        assert msg["request_id"] == 40
        assert msg["layer"] == 1
        assert (msg["px"], msg["py"], msg["pz"]) == (4, 5, 6)

    def test_list_layers(self):
        data = struct.pack("<BI", MSG_LIST_LAYERS, 50)
        msg = parse_message(data)
        assert msg["type"] == MSG_LIST_LAYERS
        assert msg["request_id"] == 50

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
        result = encode_patch_data(5, 0, 3, 1, 2, 3, None)
        assert result[0] == MSG_PATCH_DATA
        _, req_id, layer, level, px, py, pz = struct.unpack_from("<BIIBiii", result, 0)
        assert req_id == 5
        assert layer == 0
        assert level == 3
        assert (px, py, pz) == (1, 2, 3)
        encoding = result[22]
        assert encoding == ENC_EMPTY

    def test_dense_patch(self):
        data = np.full((8, 8, 8), 42, dtype=np.uint8)
        result = encode_patch_data(7, 0, 3, 0, 0, 0, data)
        assert result[0] == MSG_PATCH_DATA
        encoding = result[22]
        assert encoding == ENC_DENSE
        voxel_bytes = result[23:]
        assert len(voxel_bytes) == 512
        assert voxel_bytes[0] == 42


class TestEncodeBatchPatchData:
    def test_batch(self):
        patches = [
            (3, 0, 0, 0, np.full((8, 8, 8), 1, dtype=np.uint8)),
            (3, 1, 0, 0, None),
        ]
        result = encode_batch_patch_data(10, 0, patches)
        assert result[0] == MSG_BATCH_PATCH_DATA
        _, req_id, layer, count = struct.unpack_from("<BIIH", result, 0)
        assert req_id == 10
        assert layer == 0
        assert count == 2


class TestEncodePatchList:
    def test_list(self):
        coords = [(0, 0, 0), (1, 2, 3)]
        result = encode_patch_list(15, 0, coords)
        assert result[0] == MSG_PATCH_LIST
        _, req_id, layer, count = struct.unpack_from("<BIIH", result, 0)
        assert req_id == 15
        assert layer == 0
        assert count == 2
        px, py, pz = struct.unpack_from("<iii", result, 11)
        assert (px, py, pz) == (0, 0, 0)
        px, py, pz = struct.unpack_from("<iii", result, 23)
        assert (px, py, pz) == (1, 2, 3)


class TestEncodePutAck:
    def test_put_ack(self):
        result = encode_put_ack(20, 0, 1, 2, 3, 1)
        assert result[0] == MSG_PUT_ACK
        _, req_id, layer, px, py, pz, status = struct.unpack_from("<BIIiiiB", result, 0)
        assert req_id == 20
        assert layer == 0
        assert (px, py, pz) == (1, 2, 3)
        assert status == 1


class TestEncodeDeleteAck:
    def test_delete_ack(self):
        result = encode_delete_ack(25, 1, 4, 5, 6, 0)
        assert result[0] == MSG_DELETE_ACK
        _, req_id, layer, px, py, pz, status = struct.unpack_from("<BIIiiiB", result, 0)
        assert req_id == 25
        assert layer == 1
        assert (px, py, pz) == (4, 5, 6)
        assert status == 0


class TestEncodeLayerList:
    def test_empty(self):
        result = encode_layer_list(30, [])
        assert result[0] == MSG_LAYER_LIST
        _, req_id, count = struct.unpack_from("<BIH", result, 0)
        assert req_id == 30
        assert count == 0

    def test_multiple_layers(self):
        result = encode_layer_list(31, [0, 1, 5])
        _, req_id, count = struct.unpack_from("<BIH", result, 0)
        assert count == 3
        l0, l1, l2 = struct.unpack_from("<III", result, 7)
        assert (l0, l1, l2) == (0, 1, 5)


class TestEncodeError:
    def test_error(self):
        result = encode_error(0, 404, "Not found")
        assert result[0] == MSG_ERROR
        _, req_id, code = struct.unpack_from("<BIH", result, 0)
        assert req_id == 0
        assert code == 404
        msg = result[7:].decode("utf-8")
        assert msg == "Not found"
