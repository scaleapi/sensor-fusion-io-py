"""Tests models.Pose"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scale_sensor_fusion_io import Pose, PosePath
from scale_sensor_fusion_io.utils.pose_helpers import transform_points, transform_path


class TestPose(unittest.TestCase):
    """Tests Pose"""

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

        cls.composites = [
            cls.pose,
            Pose.from_rt(
                Rotation.from_euler("xyz", [30, 60, 90], degrees=True), [1, 2, 3]
            ),
        ]

        cls.rotations = [
            Pose.from_transform(
                [
                    [0, -0.8660254, 0.5],
                    [0.5, 0.4330127, 0.75],
                    [-0.8660254, 0.25, 0.4330127],
                ]
            ),
            Pose.from_transform(
                [
                    [0, -0.8660254, 0.5, 0],
                    [0.5, 0.4330127, 0.75, 0],
                    [-0.8660254, 0.25, 0.4330127, 0],
                    [0, 0, 0, 1],
                ]
            ),
            Pose.from_rt(Rotation.from_quat([-0.1830127, 0.5, 0.5, 0.6830127])),
            Pose.from_rt(
                Rotation.from_euler("xyz", [30, 60, 90], degrees=True), [0, 0, 0]
            ),
            Pose.from_euler([30, 60, 90], "xyz", degrees=True),
        ]

        cls.translations = [
            Pose.from_transform([1, 2, 3]),
            Pose.from_rt(translation=[1, 2, 3]),
        ]

        cls.points = [[0, 0, 0], [1, 1, 1], [10, 3, 2]]

    def test_init(self):
        """Test object creation"""
        composite = [1, 2, 3, -0.1830127, 0.5, 0.5, 0.6830127]
        rotation = [0, 0, 0, -0.1830127, 0.5, 0.5, 0.6830127]
        translation = [1, 2, 3, 0, 0, 0, 1]

        self.assertTrue(all(t == composite for t in self.composites))
        self.assertTrue(all(t == rotation for t in self.rotations))
        self.assertTrue(all(t == translation for t in self.translations))

    def test_from_transformed_points(self):
        """Tests creating a Transform from transformed points"""
        other_points = transform_points(self.pose, self.points)
        other = Pose.from_transformed_points(self.points, other_points)
        self.assertEqual(self.pose, other)

    def test_inverse(self):
        """Test inverse"""
        transformed = transform_points(self.pose, self.points)
        inverted = transform_points(self.pose.inverse, transformed)
        self.assertTrue(np.allclose(self.points, inverted))

    def test_transform_path(self):
        """Test transform_path"""
        path = PosePath.from_rt(
            Rotation.from_euler("xyz", [[30, 60, 90], [30, 0, 15]], degrees=True),
            np.array([[1, 2, 3], [2, 3, 4]]),
        )

        transform = Pose.from_rt(
            Rotation.from_euler("xyz", [5, 10, 15], degrees=True), np.array([2, 3, 4])
        )
        transformed_path = transform_path(transform, path)

        for original, transformed in zip(
            path.as_matrix(),
            transformed_path.as_matrix(),
        ):
            expected = transform @ original
            self.assertTrue(np.allclose(expected, transformed))


if __name__ == "__main__":
    unittest.main()
