"""Tests utils.pose_helpers"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scale_sensor_fusion_io import Pose
from scale_sensor_fusion_io.utils.pose_helpers import (
    interpolate_pose,
    transform_points,
    transform_points_ignore_extra_data,
)


class TestPoseHelpers(unittest.TestCase):
    """Tests utils.pose_helpers"""

    @classmethod
    def setUpClass(cls):
        cls.pose = Pose.from_transform(
            [
                [0, -0.8660254, 0.5, 1],
                [0.5, 0.4330127, 0.75, 2],
                [-0.8660254, 0.25, 0.4330127, 3],
                [0, 0, 0, 1],
            ]
        )

        cls.points = np.array([[0, 0, 0], [1, 1, 1], [10, 3, 2]])

    def test_interpolate_poses(self):
        """Tests interpolate_poses"""
        other = Pose.from_rt(
            Rotation.from_euler("xyz", [30, 60, 90], degrees=True), np.array([0, 0, 0])
        )
        self.assertEqual(interpolate_pose(self.pose, other, 0), self.pose)
        self.assertEqual(interpolate_pose(self.pose, other, 1), other)

        halfway = interpolate_pose(self.pose, other, 0.5)
        self.assertNotEqual(halfway, self.pose)
        self.assertNotEqual(halfway, other)

    def test_transform_points(self):
        """Tests transform_points"""
        transformed = transform_points(self.pose, self.points)
        rotation = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation = np.array([1, 2, 3])
        expected = rotation.apply(self.points) + translation
        self.assertTrue(np.allclose(transformed, expected))

    def test_transform_points_ignore_extra_data(self):
        """Tests transform_points_ignore_extra_data"""
        extra_data = np.random.rand(len(self.points), 2)
        points = np.hstack((self.points, extra_data))
        transformed = transform_points_ignore_extra_data(self.pose, points)
        rotation = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation = np.array([1, 2, 3])
        expected = np.hstack((rotation.apply(self.points) + translation, extra_data))
        self.assertTrue(np.allclose(transformed, expected))


if __name__ == "__main__":
    unittest.main()
