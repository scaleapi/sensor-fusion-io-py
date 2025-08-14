import numpy as np
from scale_sensor_fusion_io.models import Points, PointsSensor, PosePath
from scale_sensor_fusion_io.validation.validate import (
    validate_points_sensor,
)
from tests.validation.validate_helpers import ErrorFilter, ValidationTestBase


class TestPoseValidation(ValidationTestBase):
    def test_valid_points_sensor(self):
        sensor = PointsSensor(
            id="test",
            points=Points(
                positions=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
                colors=np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            ),
            parent_id=None,
        )

        result = validate_points_sensor(sensor)
        self.assertIsNone(result)

    def test_bad_points(self):
        path = PosePath(
            data=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            index=[0, 1, 2, -1],
        )

        sensor = PointsSensor(
            id="test",
            points=Points(
                positions=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
                colors=np.array([[0, 1, 2], [0, 0, 0]], dtype=np.float32),
            ),
            parent_id=None,
        )

        result = validate_points_sensor(sensor)
        self.assertIsNotNone(result)

        pos_dtype_filter = ErrorFilter(
            path=[
                "points",
                "positions",
            ],
            expected_errors=["Positions must have dtype float32: is float64"],
        )
        self.assertContainsError(result.details, pos_dtype_filter)

        pos_shape_filter = ErrorFilter(
            path=[
                "points",
                "positions",
            ],
            expected_errors=["Positions must have shape (n, 3): is ((9,))"],
        )
        self.assertContainsError(result.details, pos_shape_filter)

        color_shape_filter = ErrorFilter(
            path=["points", "colors"],
            expected_errors=[
                "Colors must have dtype uint8: is float32",
            ],
        )
        self.assertContainsError(result.details, color_shape_filter)

        print(result.details)

        color_length_filter = ErrorFilter(
            path=[
                "points",
                "colors",
            ],
            expected_errors=[
                "length of colors (2) should match length of positions: (9)",
            ],
        )
        self.assertContainsError(result.details, color_length_filter)

        self.assertEqual(len(result.details), 4)
