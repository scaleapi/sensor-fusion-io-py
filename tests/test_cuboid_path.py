"""Tests models.cuboid_path"""
import unittest

import numpy as np
from scale_sensor_fusion_io.models import Pose, CuboidPath
from scipy.spatial.transform import Rotation

from scale_sensor_fusion_io.utils.pose_helpers import interpolate_pose


class TestPosePath(unittest.TestCase):
    """Tests CuboidPath"""

    def test_matrix_conversion(self):
        """Test as_matrix and from_matrix"""
        path = CuboidPath([[7, 9, 11, 31, 1, 2, 2, 3, 4]])
        other = CuboidPath.from_matrix(path.as_matrix())
        self.assertTrue(np.allclose(path.to_numpy(), other.to_numpy()))

    def test_interpolate(self):
        """Test interpolate"""
        a = Pose.from_rt(
            Rotation.from_euler("xyz", [90, 0, 0], degrees=True), [1, 2, 3]
        )
        b = Pose.from_rt(
            Rotation.from_euler("xyz", [55, 0, 0], degrees=True), [7, 9, 5]
        )

        path = CuboidPath.from_matrix(
            np.stack([a.as_transform(), b.as_transform()]), index=[0, 10]
        )

        expected = interpolate_pose(a, b, 0.5)
        actual = path.interpolate([5])

        self.assertTrue(
            np.allclose(actual.as_matrix(), expected.as_transform()[np.newaxis])
        )


if __name__ == "__main__":
    unittest.main()
