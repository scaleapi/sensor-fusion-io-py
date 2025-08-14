"""Tests utils.models_common_helpers"""

import unittest

import numpy as np

from scale_sensor_fusion_io import Point3D
from scale_sensor_fusion_io.utils.models_common_helpers import point_3ds_to_matrix


class TestModelsCommonHelpers(unittest.TestCase):
    """Tests utils.models_common_helpers"""

    def test_transform_points(self):
        """Tests point_3ds_to_matrix"""
        raw_points = np.array([[0, 0, 0], [1, 1, 1], [10, 3, 2]])
        point_3ds = point_3ds_to_matrix([Point3D(*point) for point in raw_points])

        self.assertEqual(point_3ds.shape, raw_points.shape)
        self.assertTrue(np.allclose(point_3ds, raw_points))


if __name__ == "__main__":
    unittest.main()
