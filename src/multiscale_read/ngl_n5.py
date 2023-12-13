from typing import Optional, Union
import logging

from pydantic import BaseModel, root_validator, ValidationError
import numpy as np
import zarr
import xarray as xr
import dask.array as da

from .base import MultiscaleBase

logger = logging.getLogger(__name__)


class SharedMetadata(BaseModel):
    """Optional neuroglancer-specific metadata."""

    axes: Optional[list[str]] = None
    coordinateArrays: Optional[dict[str, list[str]]] = None

    @root_validator
    def check_axes(cls, values):
        if not values.get("coordinateArrays"):
            # nothing to validate
            return values

        if values.get("axes") is None:
            # can only reach this point if coordinateArrays given
            raise ValueError("coordinateArrays given, but no axes")

        for k in values["coordinateArrays"]:
            if k not in values["axes"]:
                raise ValueError("Unknown axis")

        return values

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
        if values.get("axes") is not None:
            ndim = len(values["axes"])

        pr_ndim = len(values["pixelResolution"].dimensions)
        if ndim is None:
            ndim = pr_ndim
        elif ndim != pr_ndim:
            raise ValueError("Inconsistent dimensionality")

        for row in values["scales"]:
            if len(row) != ndim:
                raise ValueError("Inconsistent dimensionality")

        return values

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

            coord_arr = np.arange(shape[idx], dtype=float) * scale[n5_idx]
            coords.append((name, coord_arr, {"units": self.pixelResolution.unit}))
            coords.append(
                xr.DataArray(
                    coord_arr,
                    dims=(name,),
                    name=name,
                    attrs={"units": self.pixelResolution.unit},
                )
                # ).pint.quantify()
            )

        return coords

    def n_scales(self) -> int:
        return len(self.scales)

    def dim_names(self) -> list[str]:
        if self.axes is not None:
            return self.axes[::-1]
        return [f"dim_{n}" for n in range(self.ndim())]


class BigDataViewerMetadata(SharedMetadata):
    downsamplingFactors: list[list[float]]
    resolution: list[float]
    units: list[str]

    @root_validator
    def check_ndim(cls, values):
        ndim = None
        if values.get("axes") is not None:
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

        return values

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
                if coord_arr is not None:
                    coords.append(
                        xr.DataArray(
                            coord_arr,
                            dims=(name,),
                            name=name,
                        )
                    )
                    continue

            if not name:
                name = f"dim_{idx}"

            coord_arr = np.arange(shape[idx], dtype=float) * scale[n5_idx]
            coords.append(
                xr.DataArray(
                    coord_arr,
                    dims=(name,),
                    name=name,
                    attrs={"units": self.units[n5_idx]},
                )
                # ).pint.quantify()
            )

        return coords

    def n_scales(self):
        return len(self.downsamplingFactors)

    def dim_names(self) -> list[str]:
        if self.axes is not None:
            return self.axes[::-1]
        return [f"dim_{n}" for n in range(self.ndim())]


class NglN5Multiscale(MultiscaleBase):
    """Neuroglancer-compatible N5 multiscale dataset.

    Use this for N5 scale pyramids with either BigDataViewer or n5-viewer metadata,
    optionally with neuroglancer extension metadata.
    """

    def __init__(self, group: zarr.Group) -> None:
        """
        Parameters
        ----------
        group : zarr.Group
            Group from N5 store with either BigDataViewer or n5-viewer
            multiscale metadata, and N datasets named s0, s1, s2, ..., s{N-1}.

        Raises
        ------
        ValueError
            If backing store is not N5.
        ValidationError
            If metadata is not compatible with either BigDataViewer or n5-viewer
        """
        self.group: zarr.Group = group
        if not isinstance(self.group.store, (zarr.N5FSStore, zarr.N5Store)):
            raise ValueError("NglN5Multiscale only supported for N5 stores")

        self.metadata: Union[BigDataViewerMetadata, N5ViewerMetadata]

        try:
            self.metadata = BigDataViewerMetadata.parse_obj(self.group.attrs)
            logger.debug("Found valid BigDataViewer metadata")
        except ValidationError:
            self.metadata = N5ViewerMetadata.parse_obj(self.group.attrs)
            logger.debug("Found valid N5ViewerMetadata")

    @classmethod
    def from_paths(cls, container, group: str, store_kwargs=None, group_kwargs=None):
        if store_kwargs is None:
            store_kwargs = dict()
        if group_kwargs is None:
            group_kwargs = dict()
        store = zarr.N5FSStore(container, **store_kwargs)
        group_kwargs.setdefault("mode", "r")
        root = zarr.open_group(store, **group_kwargs)
        return cls(root[group])

    def __len__(self) -> int:
        return self.metadata.n_scales()

    def _get_item(self, idx: int) -> xr.DataArray:
        super()._get_item(idx)
        arr = self.group[f"s{idx}"]
        coords = self.metadata.to_coords(idx, arr.shape)
        d_arr = da.from_zarr(arr)
        return xr.DataArray(d_arr, coords, name=arr.name, attrs=arr.attrs)

    def ndim(self) -> int:
        return self.metadata.ndim()
