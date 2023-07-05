import zarr
from xarray_ome_ngff import transforms_to_coords
from pydantic_ome_ngff.latest import multiscales, coordinateTransformations
import xarray as xr
import dask.array as da

# required to quantify coordinate arrrays
import pint_xarray  # noqa

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
    """Make coordinates unit-aware where possible.

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
        if "unit" in c.attrs:
            c.attrs["units"] = c.attrs["unit"]
            del c.attrs["unit"]

        if "units" in c.attrs:
            c = c.pint.quantify()
        out.append(c)
    return out


def reverse_coordinate_transformation(
    coord_trans: coordinateTransformations.CoordinateTransform, inplace=True
):
    """Reverse the dimensions of a CoordinateTransform."""
    if not inplace:
        coord_trans = coord_trans.copy(deep=True)

    if isinstance(coord_trans, coordinateTransformations.VectorScaleTransform):
        coord_trans.scale.reverse()
    elif isinstance(coord_trans, coordinateTransformations.VectorTranslationTransform):
        coord_trans.translation.reverse()

    return coord_trans


def reverse_multiscale(multiscale: multiscales.Multiscale, inplace=True):
    """Reverse the dimensions of a Multiscale."""
    if not inplace:
        multiscale = multiscale.copy(deep=True)

    multiscale.axes.reverse()

    if multiscale.coordinateTransformations is not None:
        for ct in multiscale.coordinateTransformations:
            reverse_coordinate_transformation(ct)

    for dataset in multiscale.datasets:
        for ct in dataset.coordinateTransformations:
            reverse_coordinate_transformation(ct)

    return multiscale


def reverse_multiscale_attrs(
    multiscale_attrs: multiscales.MultiscaleAttrs, inplace=True
):
    """Reverse the dimensions of a MultiscaleAttrs."""
    if not inplace:
        multiscale_attrs = multiscale_attrs.copy(deep=True)

    for mscale in multiscale_attrs.multiscales:
        reverse_multiscale(mscale)

    return multiscale_attrs
