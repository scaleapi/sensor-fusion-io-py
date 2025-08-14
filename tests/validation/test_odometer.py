from scale_sensor_fusion_io.models import PosePath, OdometrySensor
from scale_sensor_fusion_io.validation.validate import (
    validate_odometer,
)

from tests.validation.validate_helpers import ErrorFilter, ValidationTestBase


class TestPoseValidation(ValidationTestBase):
    def test_valid_odometry_sensor(self):
        path = PosePath(
            data=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            index=[0, 1, 2, 3],
        )

        sensor = OdometrySensor(
            id="test",
            poses=path,
            parent_id=None,
        )

        result = validate_odometer(sensor)
        self.assertIsNone(result)

    def test_bad_timestamps(self):
        path = PosePath(
            data=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            index=[0, 1, 2, -1],
        )

        sensor = OdometrySensor(
            id="test",
            poses=path,
            parent_id=None,
        )

        result = validate_odometer(sensor)
        self.assertIsNotNone(result)

        neg_error_filter = ErrorFilter(
            path=[
                "poses",
                "timestamps",
            ],
            expected_errors=["timestamps must not be negative"],
        )

        ts_mono_increasing_filter = ErrorFilter(
            path=[
                "poses",
                "timestamps",
            ],
            expected_errors=["timestamps must be strictly increasing"],
        )

        all_error_filters = [
            neg_error_filter,
            ts_mono_increasing_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))
