# multiscale_read

Read multiscale chunked arrays with various metadata standards as [`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html#xarray.DataArray)s.

Supported metadata:

- [x] [OME-NGFF](https://ngff.openmicroscopy.org/latest/#multiscale-md)
- [x] [Neuroglancer-N5](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/n5), which is a superset of either
  - [x] BigDataViewer
  - [x] N5Viewer

## Example usage

```python
import zarr

from multiscale_read import OmeMultiscale, NglN5Multiscale

# must have OME-NGFF metadata present
ome = OmeMultiscale.from_paths("path/to/hierarchy.zarr", "name/of/multiscale/group")
ome[0]  # base scale level

# must have Neuroglancer/ BDV/ N5V metadata present
n5 = NglN5Multiscale.from_paths("path/to/hierarchy.n5", "name/of/multiscale/group")
n5[0]  # base scale level
```

The returned `xarray.DataArray`s have coordinates in world space,
with units where possible.
They wrap over `dask.Array`s for efficient access and parallelisation.
Any attributes on the underlying zarr/ N5 array are also present in the output's `.attrs`.

Note that N5 uses a column-major convention when reporting axis metadata,
where python packages (including zarr-python's N5 implementation) use row-major.
This package reverses the required metadata when constructing the coordinate arrays,
but any attributes in `.attrs` are presented raw.
