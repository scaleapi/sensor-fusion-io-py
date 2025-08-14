import numpy as np

from scale_sensor_fusion_io.models import (
    PosePath,
    RadarSensor,
    RadarSensorFrame,
    RadarSensorPoints,
)
from scale_sensor_fusion_io.validation.validate import (
    validate_radar,
)

from tests.validation.validate_helpers import ErrorFilter, ValidationTestBase

good_pose = PosePath(
    data=[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    index=[0, 1, 2, 3],
)

good_frame = RadarSensorFrame(
    timestamp=0,
    points=RadarSensorPoints(
        positions=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        directions=np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        lengths=np.array([0, 0, 0], dtype=np.float32),
        timestamps=np.array([0, 0, 0], dtype=np.uint32),
    ),
)


class TestPoseValidation(ValidationTestBase):
    def test_valid_radar_sensor(self):
        sensor = RadarSensor(
            id="test",
            frames=[
                good_frame,
            ],
            poses=good_pose,
            parent_id=None,
        )

        result = validate_radar(sensor)
        self.assertIsNone(result)

    def test_bad_points(self):
        sensor = RadarSensor(
            id="test",
            poses=good_pose,
            frames=[
                RadarSensorFrame(
                    timestamp=0,
                    points=RadarSensorPoints(
                        positions=np.array(
                            [0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64
                        ),
                        directions=np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int16),
                        lengths=np.array([-1, 0, 257], dtype=np.int16),
                        timestamps=np.array([0, -1, -2], dtype=np.int16),
                    ),
                )
            ],
            parent_id=None,
        )

        result = validate_radar(sensor)
        self.assertIsNotNone(result)

        pos_dtype_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "positions",
            ],
            expected_errors=["Positions must have dtype float32: is float64"],
        )

        pos_shape_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "positions",
            ],
            expected_errors=["Positions must have shape (n, 3): is ((9,))"],
        )

        direction_shape_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "directions",
            ],
            expected_errors=[
                "Directions must have dtype float32: is int16",
            ],
        )

        direction_pos_size_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "directions",
            ],
            expected_errors=[
                "length of directions (2) should match length of positions: (9)"
            ],
        )

        length_shape_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "lengths",
            ],
            expected_errors=[
                "length of lengths (3) should match length of positions: (9)"
            ],
        )

        length_dtype_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "lengths",
            ],
            expected_errors=["Lengths must have dtype float32: is int16"],
        )

        timestamp_shape_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "timestamps",
            ],
            expected_errors=[
                "length of timestamps (3) should match length of positions: (9)"
            ],
        )

        timestamp_dtype_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "timestamps",
            ],
            expected_errors=["Timestamps must have dtype uint32 or uint64: is int16"],
        )

        timestamp_value_filter = ErrorFilter(
            path=[
                "frames",
                0,
                "points",
                "timestamps",
            ],
            expected_errors=[
                "frame timestamp (0) must be less than min point timestamp (-2)",
            ],
        )

        all_error_filters = [
            pos_dtype_filter,
            pos_shape_filter,
            direction_shape_filter,
            direction_pos_size_filter,
            length_shape_filter,
            length_dtype_filter,
            timestamp_shape_filter,
            timestamp_dtype_filter,
            timestamp_value_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))

    def test_bad_pose(self):
        bad_pose = PosePath(
            data=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            index=[0, 1, 2, -1],
        )
        sensor = RadarSensor(
            id="test",
            poses=bad_pose,
            frames=[
                good_frame,
            ],
            parent_id=None,
        )

        result = validate_radar(sensor)

        print(result.details)
        self.assertIsNotNone(result)

        neg_error_filter = ErrorFilter(
            path=["poses", "timestamps"],
            expected_errors=["timestamps must not be negative"],
        )

        ts_mono_increasing_filter = ErrorFilter(
            path=["poses", "timestamps"],
            expected_errors=["timestamps must be strictly increasing"],
        )

        all_error_filters = [
            neg_error_filter,
            ts_mono_increasing_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))

    def test_bad_fps(self):
        sensor = RadarSensor(
            id="test",
            poses=good_pose,
            frames=[
                good_frame,
                RadarSensorFrame(
                    timestamp=1,
                    points=RadarSensorPoints(
                        positions=np.array(
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32
                        ),
                        directions=np.array(
                            [[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=np.float32
                        ),
                        lengths=np.array([0, 0, 0], dtype=np.float32),
                        timestamps=np.array([1, 1, 1], dtype=np.uint32),
                    ),
                ),
            ],
            parent_id=None,
        )

        result = validate_radar(sensor)

        print(result.details)
        self.assertIsNotNone(result)

        fps_filter = ErrorFilter(
            path=["frames"],
            expected_errors=["approximate fps is too high: 1000000.0 > 100"],
        )

        all_error_filters = [
            fps_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))

    def test_bad_consecutive_frames(self):
        sensor = RadarSensor(
            id="test",
            poses=good_pose,
            frames=[
                good_frame,
                RadarSensorFrame(
                    timestamp=-1,
                    points=RadarSensorPoints(
                        positions=np.array(
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32
                        ),
                        directions=np.array(
                            [[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=np.float32
                        ),
                        lengths=np.array([0, 0, 0], dtype=np.float32),
                        timestamps=np.array([1, 1, 1], dtype=np.uint32),
                    ),
                ),
            ],
            parent_id=None,
        )

        result = validate_radar(sensor)

        print(result.details)
        self.assertIsNotNone(result)

        non_neg_filter = ErrorFilter(
            path=["frames"],
            expected_errors=["timestamps must not be negative"],
        )

        strictly_increasing_ts_filter = ErrorFilter(
            path=["frames"],
            expected_errors=["timestamps must be strictly increasing"],
        )

        all_error_filters = [
            non_neg_filter,
            strictly_increasing_ts_filter,
        ]

        for error_filter in all_error_filters:
            self.assertContainsError(result.details, error_filter)

        self.assertEqual(len(result.details), len(all_error_filters))
