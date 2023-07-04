from typing import Optional, Union

from pydantic import BaseModel, root_validator, ValidationError
import numpy as np
import zarr
import xarray as xr

from .base import MultiscaleBase


class SharedMetadata(BaseModel):
    axes: Optional[list[str]]
    coordinateArrays: Optional[dict[str, list[str]]]

    @root_validator
    def check_axes(cls, values):
        if not values["coordinateArrays"]:
            return

        if values["axes"] is None:
            raise ValueError("coordinateArrays given, but no axes")

        for k in values["coordinateArrays"]:
            if k not in values["axes"]:
                raise ValueError("Unknown axis")

    def coordinate_array(self, axis: str) -> Optional[list[str]]:
        if self.axes is None or self.coordinateArrays is None:
            return None
        return self.coordinateArrays.get(axis)


class PixelResolution(BaseModel):
    unit: str
    dimensions: list[float]

    def ndim(self):
        return len(self.dimensions)


class N5ViewerMetadata(SharedMetadata):
    scales: list[list[float]]
    pixelResolution: PixelResolution

    @root_validator
    def check_ndim(cls, values):
        ndim = None
        if values["axes"] is not None:
            ndim = len(values["axes"])

        pr_ndim = len(values["pixelResolution"].dimensions)
        if ndim is None:
            ndim = pr_ndim
        elif ndim != pr_ndim:
            raise ValueError("Inconsistent dimensionality")

        for row in values["scales"]:
            if len(row) != ndim:
                raise ValueError("Inconsistent dimensionality")

    def ndim(self):
        return self.pixelResolution.ndim()

    def to_coords(self, scale_idx: int, shape: tuple[int]):
        if len(shape) != self.ndim():
            raise ValueError("Inconsistent dimensionality")
        coords = []
        scale = np.array(self.pixelResolution.dimensions) * self.scales[scale_idx]
        for idx in range(self.ndim()):
            n5_idx = self.ndim() - idx - 1
            if self.axes is None:
                name = f"dim_{idx}"
            else:
                name = self.axes[n5_idx]
                coord_arr = self.coordinate_array(name)
                if coords is not None:
                    coords.append((name, coord_arr))
                    continue

                if not name:
                    name = f"dim_{idx}"

            coord_arr = np.arange(shape[idx], float) * scale[n5_idx]
            coords.append((name, coord_arr, {"units": self.pixelResolution.unit}))

    def n_scales(self) -> int:
        return len(self.scales)


class BigDataViewerMetadata(SharedMetadata):
    downsamplingFactors: list[list[float]]
    resolution: list[float]
    units: list[str]

    @root_validator
    def check_ndim(cls, values):
        ndim = None
        if values["axes"] is not None:
            ndim = len(values["axes"])

        pr_ndim = len(values["resolution"])
        if ndim is None:
            ndim = pr_ndim
        elif ndim != pr_ndim:
            raise ValueError("Inconsistent dimensionality")

        if ndim != len(values["units"]):
            raise ValueError("Inconsistent dimensionality")

        for row in values["downsamplingFactors"]:
            if len(row) != ndim:
                raise ValueError("Inconsistent dimensionality")

    def ndim(self):
        return len(self.resolution)

    def to_coords(self, scale_idx: int, shape: tuple[int]):
        if len(shape) != self.ndim():
            raise ValueError("Inconsistent dimensionality")
        coords = []
        scale = np.array(self.resolution) * self.downsamplingFactors[scale_idx]
        for idx in range(self.ndim()):
            n5_idx = self.ndim() - idx - 1
            if self.axes is None:
                name = f"dim_{idx}"
            else:
                name = self.axes[n5_idx]
                coord_arr = self.coordinate_array(name)
                if coords is not None:
                    coords.append((name, coord_arr))
                    continue

                if not name:
                    name = f"dim_{idx}"

            coord_arr = np.arange(shape[idx], float) * scale[n5_idx]
            coords.append((name, coord_arr, {"unit": self.units[n5_idx]}))

    def n_scales(self):
        return len(self.downsamplingFactors)


class NglN5Multiscale(MultiscaleBase):
    def __init__(self, group: zarr.Group) -> None:
        self.group: zarr.Group = group
        if not isinstance(self.group.store, (zarr.N5FSStore, zarr.N5Store)):
            raise ValueError("NglN5Multiscale only supported for N5 stores")

        try:
            meta = BigDataViewerMetadata.parse_obj(self.group.attrs)
        except ValidationError:
            meta = N5ViewerMetadata.parse_obj(self.group.attrs)

        self.metadata: Union[BigDataViewerMetadata, N5ViewerMetadata] = meta

    def __len__(self) -> int:
        return self.metadata.n_scales()

    def __contains__(self, idx: int) -> bool:
        return -len(self) < idx < len(self)

    def __getitem__(self, idx: int) -> xr.DataArray:
        arr = self.group[f"s{idx}"]
        coords = self.metadata.to_coords(idx, arr.shape)
        return xr.DataArray(arr, coords, name=arr.name, attrs=arr.attrs)
