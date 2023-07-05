from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

import xarray as xr


class MultiscaleBase(Sequence, ABC):
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def _get_item(self, idx: int):
        if not (-len(self) < idx < len(self)):
            raise IndexError("index out of range")

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[xr.DataArray, list[xr.DataArray]]:
        if isinstance(idx, int):
            return self._get_item(idx)

        out = []
        start, stop, stride = idx.indices(len(self))
        while start < stop:
            out.append(self._get_item(start))
            start += stride
        return out
