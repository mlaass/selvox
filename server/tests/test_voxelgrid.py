import numpy as np
import pytest

from voxel_server.voxelgrid import VoxelGrid


class TestDownsample:
    def test_all_zeros(self):
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        result = VoxelGrid._downsample(data)
        assert result.shape == (2, 2, 2)
        assert np.all(result == 0)

    def test_uniform_value(self):
        data = np.full((4, 4, 4), 5, dtype=np.uint8)
        result = VoxelGrid._downsample(data)
        assert result.shape == (2, 2, 2)
        assert np.all(result == 5)

    def test_max_nonzero(self):
        """Max non-zero value in the block should win."""
        data = np.zeros((2, 2, 2), dtype=np.uint8)
        data[0, 0, 0] = 3
        data[0, 0, 1] = 3
        data[0, 1, 0] = 3
        data[1, 0, 0] = 7
        result = VoxelGrid._downsample(data)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == 7

    def test_empty_with_some_values(self):
        """Non-zero value should win even if most are empty."""
        data = np.zeros((2, 2, 2), dtype=np.uint8)
        data[0, 0, 0] = 10
        result = VoxelGrid._downsample(data)
        assert result[0, 0, 0] == 10

    def test_1x1x1_passthrough(self):
        data = np.array([[[42]]], dtype=np.uint8)
        result = VoxelGrid._downsample(data)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == 42


class TestVoxelGrid:
    def test_add_and_get_patch(self):
        grid = VoxelGrid()
        data = np.zeros((64, 64, 64), dtype=np.uint8)
        data[0, 0, 0] = 1
        data[32, 32, 32] = 2
        grid.add_patch(0, 0, 0, 0, data)

        # Level 6 should be the original data
        result = grid.get_patch(0, 0, 0, 0, 6)
        assert result is not None
        assert result.shape == (64, 64, 64)
        assert result[0, 0, 0] == 1
        assert result[32, 32, 32] == 2

    def test_lod_levels_shape(self):
        grid = VoxelGrid()
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 1, 2, 3, data)

        expected_sizes = [1, 2, 4, 8, 16, 32, 64]
        for level in range(7):
            result = grid.get_patch(0, 1, 2, 3, level)
            assert result is not None
            assert result.shape == (expected_sizes[level],) * 3

    def test_missing_patch_returns_none(self):
        grid = VoxelGrid()
        assert grid.get_patch(0, 0, 0, 0, 6) is None

    def test_list_patches(self):
        grid = VoxelGrid()
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data)
        grid.add_patch(0, 1, 0, 0, data)
        grid.add_patch(0, 0, 1, 2, data)

        patches = grid.list_patches(0)
        assert len(patches) == 3
        assert (0, 0, 0) in patches
        assert (1, 0, 0) in patches
        assert (0, 1, 2) in patches

    def test_delete_patch(self):
        grid = VoxelGrid()
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data)
        assert grid.get_patch(0, 0, 0, 0, 6) is not None
        grid.delete_patch(0, 0, 0, 0)
        assert grid.get_patch(0, 0, 0, 0, 6) is None

    def test_delete_removes_empty_layer(self):
        grid = VoxelGrid()
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data)
        grid.delete_patch(0, 0, 0, 0)
        assert 0 not in grid.list_layers()

    def test_list_layers(self):
        grid = VoxelGrid()
        assert grid.list_layers() == []
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data)
        grid.add_patch(2, 0, 0, 0, data)
        grid.add_patch(5, 1, 0, 0, data)
        assert grid.list_layers() == [0, 2, 5]

    def test_multi_layer_isolation(self):
        grid = VoxelGrid()
        data0 = np.full((64, 64, 64), 1, dtype=np.uint8)
        data1 = np.full((64, 64, 64), 2, dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data0)
        grid.add_patch(1, 0, 0, 0, data1)

        # Each layer has its own data
        r0 = grid.get_patch(0, 0, 0, 0, 6)
        r1 = grid.get_patch(1, 0, 0, 0, 6)
        assert r0[0, 0, 0] == 1
        assert r1[0, 0, 0] == 2

        # list_patches only shows the layer's own patches
        assert len(grid.list_patches(0)) == 1
        assert len(grid.list_patches(1)) == 1

        # Deleting from layer 0 doesn't affect layer 1
        grid.delete_patch(0, 0, 0, 0)
        assert grid.get_patch(0, 0, 0, 0, 6) is None
        assert grid.get_patch(1, 0, 0, 0, 6) is not None

    def test_world_size(self):
        grid = VoxelGrid()
        data = np.ones((64, 64, 64), dtype=np.uint8)
        grid.add_patch(0, 0, 0, 0, data)
        assert grid.world_size == (64, 64, 64)
        grid.add_patch(0, 3, 1, 2, data)
        assert grid.world_size == (256, 128, 192)
