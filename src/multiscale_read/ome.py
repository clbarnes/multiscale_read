import zarr
from xarray_ome_ngff import transforms_to_coords
from pydantic_ome_ngff.latest import multiscales
import xarray as xr
import dask.array as da

from .utils import UNITS_ATTR, OTHER_UNITS_ATTR
from .base import MultiscaleBase


class OmeMultiscale(MultiscaleBase):
    """OME-NGFF multiscale dataset."""

    def __init__(self, group: zarr.Group, index=0) -> None:
        """
        Parameters
        ----------
        group : zarr.Group
            Group containing multiscale metadata.
        index : int, optional
            Groups can contain several scale pyramids.
            This selects which one to open (default 0).
        """
        self.group: zarr.Group = group
        mgrp = multiscales.MultiscaleAttrs.parse_obj(self.group.attrs)
        self.multiscales: multiscales.Multiscale = mgrp.multiscales[index]

    @classmethod
    def from_paths(cls, container, group=None, index=0):
        grp = zarr.open_group(container, mode="r", path=group)
        return cls(grp, index)

    def __len__(self) -> int:
        return len(self.multiscales.datasets)

    def _get_item(self, idx: int) -> xr.DataArray:
        super()._get_item(idx)
        ds = self.multiscales.datasets[idx]
        transforms = ds.coordinateTransformations
        arr = self.group[ds.path]
        coords = transmute_coords(
            transforms_to_coords(
                self.multiscales.axes,
                transforms,
                arr.shape,
            )
        )
        d_arr = da.from_zarr(arr)

        return xr.DataArray(d_arr, coords, name=arr.name, attrs=arr.attrs)

    def ndim(self):
        return len(self.multiscales.axes)


def transmute_coords(coords: list[xr.DataArray]) -> list[xr.DataArray]:
    """Update the name of the attribute containing units.

    Parameters
    ----------
    coords : list[xr.DataArray]

    Returns
    -------
    list[xr.DataArray]
    """
    out = []
    for c in coords:
        # shim for https://github.com/JaneliaSciComp/xarray-ome-ngff/issues/2
        if OTHER_UNITS_ATTR in c.attrs:
            c.attrs[UNITS_ATTR] = c.attrs[OTHER_UNITS_ATTR]
            del c.attrs[OTHER_UNITS_ATTR]

        # if "units" in c.attrs:
        #     c = c.pint.quantify()
        out.append(c)
    return out
