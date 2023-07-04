from collections.abc import Iterator

import zarr
from xarray_ome_ngff import transforms_to_coords
from pydantic_ome_ngff.latest import multiscales
import xarray as xr

from .base import MultiscaleBase


class OmeMultiscale(MultiscaleBase):
    def __init__(self, group: zarr.Group, index=0) -> None:
        self.group: zarr.Group = group
        mgrp = multiscales.MultiscaleAttrs.parse_obj(self.group.attrs)
        self.multiscales: multiscales.Multiscale = mgrp.multiscales[index]

    def __iter__(self) -> Iterator[int]:
        for idx, _ in enumerate(self.multiscales.datasets):
            yield idx

    def __len__(self) -> int:
        return len(self.multiscales.datasets)

    def __contains__(self, idx: int) -> bool:
        return -len(self) < idx < len(self)

    def __getitem__(self, idx: int) -> xr.DataArray:
        ds = self.multiscales.datasets[idx]
        transforms = ds.coordinateTransformations
        arr = self.group[ds.path]
        coords = transforms_to_coords(
            self.multiscales.axes,
            transforms,
            arr.shape,
        )
        return xr.DataArray(arr, coords, name=arr.name, attrs=arr.attrs)
