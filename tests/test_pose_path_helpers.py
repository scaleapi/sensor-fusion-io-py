"""Test utils.pose_path_helpers"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scale_sensor_fusion_io import PosePath
from scale_sensor_fusion_io.utils.pose_path_helpers import (
    apply_interpolated_transform_to_points,
    apply_pose_path_to_points,
    apply_pose_path_to_single_point,
)


class TestPosePathHelpers(unittest.TestCase):
    """Test utils.pose_path_helpers"""

    def test_apply_interpolated_transform_to_points(self):
        """Test apply_interpolated_transform_to_points with resolution=2"""
        data = np.array(
            [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1]]
        )
        index = [0, 2, 4]

        pose_path = PosePath(data, index=index)
        points = np.array([[0, 0, 0], [1, 1, 1]])
        timestamps = np.array([1, 5])

        transformed_points = apply_interpolated_transform_to_points(
            points, timestamps, pose_path, resolution=2
        )
        expected_points = np.array([[0, 0, 0], [2, 2, 2]])

        self.assertTrue(np.allclose(transformed_points, expected_points))

    def test_apply_interpolated_transform_to_points_wo_resolution(self):
        """Test apply_interpolated_transform_to_points with resolution=None"""
        data = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1]]
        index = [0, 1, 2]

        pose_path = PosePath(data, index=index)
        points = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        timestamps = [0.5, 1, 1.5, 2, 2.5]

        transformed_points = apply_interpolated_transform_to_points(
            points, timestamps, pose_path
        )

        expected_points = np.array(
            [[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [1, 1, 1]]
        )

        self.assertTrue(np.allclose(transformed_points, expected_points))

    def test_apply_interpolated_transform_to_points_unordered_timestamps(self):
        """Test apply_interpolated_transform_to_points with timestamps out of order"""
        data = [[1, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
        index = [2, 0, 1]

        pose_path = PosePath(data, index=index)
        points = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        timestamps = [1.5, 2, 1, 0.5, 2.5]

        transformed_points = apply_interpolated_transform_to_points(
            points, timestamps, pose_path
        )

        expected_points = np.array(
            [[0.5, 0.5, 0.5], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]]
        )

        self.assertTrue(np.allclose(transformed_points, expected_points))

    def test_apply_interpolated_transform_to_points_wo_resolution_w_extrapolation(self):
        """Test apply_interpolated_transform_to_points with extrapolation_strategy='linear'"""
        data = np.array(
            [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1]]
        )
        index = [0, 2, 4]

        pose_path = PosePath(data, index=index)
        points = np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ).astype(np.float32)
        timestamps = np.array([1, 2, 3, 4, 5])

        transformed_points = apply_interpolated_transform_to_points(
            points, timestamps, pose_path, resolution=1, extrapolation_strategy="linear"
        )

        expected_points = np.array(
            [[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [1.5, 1.5, 1.5]]
        )

        self.assertTrue(np.allclose(transformed_points, expected_points))

    def test_apply_pose_path_to_points(self):
        """Test apply_pose_path_to_points"""
        rotation = Rotation.from_euler("xyz", [[30, 60, 90], [30, 0, 15]], degrees=True)
        translation = np.array([[1, 2, 3], [2, 3, 4]])
        path = PosePath.from_rt(rotation, translation)

        points = [[1, 1, 1], [4, 6, 5]]
        transformed_points = apply_pose_path_to_points(path, points)
        expected_points = rotation.apply(points) + translation
        self.assertTrue(np.allclose(transformed_points, expected_points))

    def test_apply_pose_path_to_single_point(self):
        """Test apply_pose_path_to_single_point"""
        rotation = Rotation.from_euler("xyz", [[30, 60, 90], [30, 0, 15]], degrees=True)
        translation = np.array([[1, 2, 3], [2, 3, 4]])
        path = PosePath.from_rt(rotation, translation)

        point = [1, 1, 1]
        transformed_points = apply_pose_path_to_single_point(path, point)
        expected_points = rotation.apply(point) + translation
        self.assertTrue(np.allclose(transformed_points, expected_points))


if __name__ == "__main__":
    unittest.main()
