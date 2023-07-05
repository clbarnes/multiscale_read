from typing import Hashable, NamedTuple, Optional
import xarray as xr
import numpy as np
from pydantic_ome_ngff.latest import coordinateTransformations, multiscales

UNITS_ATTR = "unit"
OTHER_UNITS_ATTR = "units"


def reverse_coordinate_transformation(
    coord_trans: coordinateTransformations.CoordinateTransform, inplace=True
):
    """Reverse the dimensions of a CoordinateTransform.

    e.g. for switching between N5 and Zarr dimension order conventions.
    """
    if not inplace:
        coord_trans = coord_trans.copy(deep=True)

    if isinstance(coord_trans, coordinateTransformations.VectorScaleTransform):
        coord_trans.scale.reverse()
    elif isinstance(coord_trans, coordinateTransformations.VectorTranslationTransform):
        coord_trans.translation.reverse()

    return coord_trans


def reverse_multiscale(multiscale: multiscales.Multiscale, inplace=True):
    """Reverse the dimensions of a Multiscale.

    e.g. for switching between N5 and Zarr dimension order conventions.
    """
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
    """Reverse the dimensions of a MultiscaleAttrs.

    e.g. for switching between N5 and Zarr dimension order conventions.
    """
    if not inplace:
        multiscale_attrs = multiscale_attrs.copy(deep=True)

    for mscale in multiscale_attrs.multiscales:
        reverse_multiscale(mscale)

    return multiscale_attrs


DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8


class ArrayInfo(NamedTuple):
    offset: dict[Hashable, Optional[float]]
    resolution: dict[Hashable, Optional[float]]
    units: dict[Hashable, Optional[str]]
    shape: dict[Hashable, int]
    order: list[Hashable]

    def _ordered(self, d: dict):
        return [d[o] for o in self.order]

    def ordered_offset(self):
        return self._ordered(self.offset)

    def ordered_resolution(self):
        return self._ordered(self.resolution)

    def ordered_units(self):
        return self._ordered(self.units)

    def ordered_shape(self):
        return self._ordered(self.shape)

    def reverse_order(self):
        return type(self)(self.offset, self.resolution, self.units, self.order[::-1])

    @classmethod
    def from_xarray(
        cls, arr: xr.DataArray, rel_abs_tolerances: Optional[tuple[float, float]] = None
    ):
        offset = dict()
        resolution = dict()
        units = dict()
        shape = dict()
        order = []

        for d, c_arr in arr.coords.items():
            order.append(d)
            shape[d] = len(c_arr)
            units[d] = c_arr.attrs.get(UNITS_ATTR)

            if not np.issubdtype(c_arr.dtype, np.number):
                offset[d] = None
                resolution[d] = None
                continue

            offset[d] = c_arr[0]

            if rel_abs_tolerances is None:
                res = c_arr[1] - c_arr[0]
            else:
                diffs = np.unique(np.diff(c_arr))
                rtol, atol = rel_abs_tolerances
                if rtol is None:
                    rtol = DEFAULT_RTOL
                if atol is None:
                    atol = DEFAULT_ATOL

                if diffs.ptp() > atol + np.abs(diffs.min()) * rtol:
                    raise ValueError("Resolution is inconsistent")

                res = diffs[0]

            if res <= 0:
                raise ValueError("Resolution is not monotonically increasing")

            resolution[d] = res

        return cls(offset, resolution, units, order)
