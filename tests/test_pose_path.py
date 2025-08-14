import unittest

import numpy as np
import pandas as pd
from scale_sensor_fusion_io.models import Pose, PosePath
from scipy.spatial.transform import Rotation

from scale_sensor_fusion_io.utils.pose_helpers import interpolate_pose


class TestPosePath(unittest.TestCase):
    def test_from_rt(self):
        rotation = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation = np.array([1, 2, 3])
        path = PosePath.from_rt(rotation, translation)

        self.assertTrue(np.allclose(path.positions, translation))
        self.assertTrue(np.allclose(path.headings, rotation.as_quat()))

    def test_from_euler(self):
        angles = np.array([[30, 60, 90]])
        path = PosePath.from_euler("xyz", angles, degrees=True)
        expected_rotation = Rotation.from_euler("xyz", angles, degrees=True)
        self.assertTrue(np.allclose(path.headings, expected_rotation.as_quat()))

    def test_as_euler(self):
        angles = np.array([[30, 60, 90]])
        path = PosePath.from_euler("xyz", angles, degrees=True)
        result = path.as_euler("xyz", degrees=True)
        self.assertTrue(np.allclose(result, angles))

    def test_from_positions(self):
        positions = np.array([[1, 2, 3]])
        path = PosePath.from_positions(positions)
        self.assertTrue(np.allclose(path.positions, positions))
        self.assertTrue(np.allclose(path.headings, np.array([[0, 0, 0, 1]])))

    def test_invert(self):
        rotation = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation = np.array([1, 2, 3])
        path = PosePath.from_rt(rotation, translation)
        inverted_path = path.invert()
        identity_path = path @ inverted_path

        self.assertTrue(np.allclose(identity_path.positions, np.zeros((1, 3))))
        self.assertTrue(np.allclose(identity_path.headings, np.array([[0, 0, 0, 1]])))

    def test_matmul(self):
        rotation1 = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation1 = np.array([1, 2, 3])
        path1 = PosePath.from_rt(rotation1, translation1)

        rotation2 = Rotation.from_euler("xyz", [45, 45, 45], degrees=True)
        translation2 = np.array([4, 5, 6])
        path2 = PosePath.from_rt(rotation2, translation2)

        result_path = path1 @ path2
        expected_matrix = path1.as_matrix() @ path2.as_matrix()
        self.assertTrue(np.allclose(result_path.as_matrix(), expected_matrix))

    def test_make_relative(self):
        rotation1 = Rotation.from_euler("xyz", [30, 60, 90], degrees=True)
        translation1 = np.array([1, 2, 3])
        path1 = PosePath.from_rt(rotation1, translation1)

        rotation2 = Rotation.from_euler("xyz", [45, 45, 45], degrees=True)
        translation2 = np.array([4, 5, 6])
        path2 = PosePath.from_rt(rotation2, translation2)

        combined_path = PosePath(np.vstack([path1.values, path2.values]))
        relative_path = combined_path.make_relative()

        self.assertTrue(np.allclose(relative_path[:1].positions, np.zeros((1, 3))))
        self.assertTrue(
            np.allclose(relative_path[:1].headings, np.array([[0, 0, 0, 1]]))
        )

        expected_relative_matrix = np.linalg.inv(path1.as_matrix()) @ path2.as_matrix()
        expected_relative_path = PosePath.from_matrix(expected_relative_matrix)
        self.assertTrue(
            np.allclose(relative_path[1:].positions, expected_relative_path.positions)
        )
        self.assertTrue(
            np.allclose(relative_path[1:].headings, expected_relative_path.headings)
        )

    def test_interpolate(self):
        """Test interpolate"""
        a = Pose.from_rt(
            Rotation.from_euler("xyz", [90, 60, 90], degrees=True), [1, 2, 3]
        )
        b = Pose.from_rt(
            Rotation.from_euler("xyz", [55, 30, 10], degrees=True), [7, 9, 5]
        )
        path = PosePath.from_matrix(
            np.stack([a.as_transform(), b.as_transform()]), index=[0, 10]
        )

        expected = interpolate_pose(a, b, 0.5)
        actual = path.interpolate([5])

        self.assertTrue(
            np.allclose(actual.as_matrix(), expected.as_transform()[np.newaxis])
        )

    def test_calculate_kinematics_uniform_motion_linear_time(self):
        timestamps = np.array([0, 1000000, 2000000, 3000000])
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        path = PosePath.from_positions(positions)
        path.index = timestamps

        # Time unit is in microseconds here
        kinematics = path.calculate_kinematics(time_unit_to_second_conversion=1e6)

        expected_kinematics = pd.DataFrame({
            'vx': [1.0, 1.0, 1.0, 1.0],
            'vy': [1.0, 1.0, 1.0, 1.0],
            'vz': [1.0, 1.0, 1.0, 1.0],
            'ax': [0.0, 0.0, 0.0, 0.0],
            'ay': [0.0, 0.0, 0.0, 0.0],
            'az': [0.0, 0.0, 0.0, 0.0],
        }, index=[0.0, 1.0, 2.0, 3.0])

        pd.testing.assert_frame_equal(kinematics, expected_kinematics)

    def test_calculate_kinematics_parabolic_motion_linear_time(self):
        timestamps = np.array([0, 1000000000, 2000000000, 3000000000])
        positions = np.array([[0, 0, 0], [1, 1, 1], [4, 2, 2], [9, 3, 3]])
        path = PosePath.from_positions(positions)
        path.index = timestamps

        kinematics = path.calculate_kinematics()

        expected_kinematics = pd.DataFrame({
            'vx': [0.0, 2.0, 4.0, 6.0],
            'vy': [1.0, 1.0, 1.0, 1.0],
            'vz': [1.0, 1.0, 1.0, 1.0],
            'ax': [2.0, 2.0, 2.0, 2.0],
            'ay': [0.0, 0.0, 0.0, 0.0],
            'az': [0.0, 0.0, 0.0, 0.0],
        }, index=[0.0, 1.0, 2.0, 3.0])

        pd.testing.assert_frame_equal(kinematics, expected_kinematics)

    def test_calculate_kinematics_uniform_motion_variable_time(self):
        timestamps = np.array([0, 2000000000, 5000000000, 9000000000])
        positions = np.array([[0, 0, 0], [2, 2, 2], [5, 5, 5], [9, 9, 9]])
        path = PosePath.from_positions(positions)
        path.index = timestamps

        kinematics = path.calculate_kinematics()

        expected_kinematics = pd.DataFrame({
            'vx': [1.0, 1.0, 1.0, 1.0],
            'vy': [1.0, 1.0, 1.0, 1.0],
            'vz': [1.0, 1.0, 1.0, 1.0],
            'ax': [0.0, 0.0, 0.0, 0.0],
            'ay': [0.0, 0.0, 0.0, 0.0],
            'az': [0.0, 0.0, 0.0, 0.0],
        }, index=[0.0, 2.0, 5.0, 9.0])

        pd.testing.assert_frame_equal(kinematics, expected_kinematics)

    def test_calculate_kinematics_parabolic_motion_variable_time(self):
        timestamps = np.array([0, 1000000000, 4000000000, 8000000000])
        positions = np.array([[0, 0, 0], [1, 1, 0], [16, 4, 0], [64, 8, 0]])
        path = PosePath.from_positions(positions)
        path.index = timestamps

        kinematics = path.calculate_kinematics()

        expected_kinematics = pd.DataFrame({
            'vx': [0.0, 2.0, 8.0, 16.0],
            'vy': [1.0, 1.0, 1.0, 1.0],
            'vz': [0.0, 0.0, 0.0, 0.0],
            'ax': [2.0, 2.0, 2.0, 2.0],
            'ay': [0.0, 0.0, 0.0, 0.0],
            'az': [0.0, 0.0, 0.0, 0.0],
        }, index=[0.0, 1.0, 4.0, 8.0])

        pd.testing.assert_frame_equal(kinematics, expected_kinematics)


if __name__ == "__main__":
    unittest.main()
