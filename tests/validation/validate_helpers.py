from typing import Callable, List, Optional
import unittest

from scale_sensor_fusion_io.validation.error import ErrorDetails, PathField


ErrorFilterFn = Callable[[ErrorDetails], bool]


class ErrorFilter:
    path: List[PathField]
    expected_errors: List[str]

    def __init__(self, path: List[PathField], expected_errors: List[str]):
        self.path = path
        self.expected_errors = expected_errors

    def apply(self, error_details: ErrorDetails) -> bool:
        return (
            error_details.path == self.path
            and error_details.errors == self.expected_errors
        )

    def __repr__(self) -> str:
        return f"Filter<path={self.path}, expected_errors={self.expected_errors}>"


class ValidationTestBase(unittest.TestCase):
    def assertContainsError(
        self,
        error_details: List[ErrorDetails],
        filter: ErrorFilter,
        msg: Optional[str] = None,
    ) -> None:
        errors_for_path = [
            e for ed in error_details if ed.path == filter.path for e in ed.errors
        ]

        if len(errors_for_path) == 0 or any(
            e not in errors_for_path for e in filter.expected_errors
        ):
            self.fail(
                self._formatMessage(
                    msg,
                    f"Error details did not contain all expected errors {filter.expected_errors} for path {filter.path}. Contained {errors_for_path}",
                )
            )
