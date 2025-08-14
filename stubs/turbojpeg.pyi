import array as _array
import mmap
from ctypes import Structure, c_int
from typing import Any, Iterable, Iterator, Optional, TypeVar, Union
import numpy as np
import numpy.typing as npt

_SupportsBuffer = Union[
    bytes,
    bytearray,
    memoryview,
    _array.array[Any],
    mmap.mmap,
    npt.NDArray[Any],
    np.generic,
]


class ScalingFactor(Structure):
    _fields_ = ('num', c_int), ('denom', c_int)


TJPF_BGR = 1
TJSAMP_422 = 1

class TurboJPEG:
    def decode(self, jpeg_buf: _SupportsBuffer, pixel_format: int = TJPF_BGR,
               scaling_factor: Optional[ScalingFactor] = None, flags: int = 0) -> npt.NDArray[np.uint8]: ...

    def encode(self, img: npt.NDArray[np.uint8], quality: int = 100, pixel_format: int = TJPF_BGR, jpeg_subsample: int = TJSAMP_422,
               flags: int = 0) -> bytes: ...