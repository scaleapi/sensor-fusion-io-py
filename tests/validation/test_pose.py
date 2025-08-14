from scale_sensor_fusion_io.models import PosePath
from scale_sensor_fusion_io.validation.validate import validate_pose_path

from tests.validation.validate_helpers import ErrorFilter, ValidationTestBase


class TestPoseValidation(ValidationTestBase):
    def test_valid_pose_path(self):
        path = PosePath(
            data=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            index=[0, 1, 2, 3],
        )

        result = validate_pose_path(path)
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

        result = validate_pose_path(path)
        print(result)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.details), 2)
        neg_error_filter = ErrorFilter(
            path=["timestamps"], expected_errors=["timestamps must not be negative"]
        )

        ts_mono_increasing_filter = ErrorFilter(
            path=["timestamps"],
            expected_errors=["timestamps must be strictly increasing"],
        )

        all_error_filters = [
            neg_error_filter,
            ts_mono_increasing_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))
