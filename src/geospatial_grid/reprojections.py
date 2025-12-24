import xarray as xr
import pyproj
import rasterio
from affine import Affine
from typing import Tuple, Dict
from geospatial_grid.gsgrid import GSGrid
import numpy as np


def reproject_data(
    data: xr.Dataset | xr.DataArray,
    new_crs: pyproj.CRS,
    new_resolution: float | None = None,
    resampling: rasterio.enums.Resampling | None = None,
    nodata: int | float | None = None,
    transform: Affine | None = None,
    shape: Tuple[int, int] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Wrapper of rioxarray reproject so that it's typed.

    See https://corteva.github.io/rioxarray/html/examples/reproject.html for params description.

    """
    # Wrap rioxarray reproject_data so that it's typed

    # Rioxarray reproject nearest by default
    return data.rio.reproject(
        dst_crs=new_crs,
        resolution=new_resolution,
        resampling=resampling,
        transform=transform,
        nodata=nodata,
        shape=shape,
    )


def reproject_using_grid(
    data: xr.Dataset | xr.DataArray,
    output_grid: GSGrid,
    nodata: int | float | None = None,
    resampling_method: rasterio.enums.Resampling | None = None,
) -> xr.Dataset | xr.DataArray:
    """Object oriented regridding function.

    Args:
        data (xr.Dataset | xr.DataArray): Data to reproject
        output_grid (GSGrid): Output grid definition in the form of an object
        nodata (int | float | None, optional): no data value of the output Xarray object. Defaults to None.
        resampling_method (rasterio.enums.Resampling | None, optional): Resampling method. Defaults to nearest in rio.reproject().

    Returns:
        xr.Dataset | xr.DataArray: the regridded Xarray object
    """
    data_reprojected = reproject_data(
        data=data,
        shape=output_grid.shape,
        transform=output_grid.affine,
        new_crs=output_grid.crs,
        resampling=resampling_method,
        nodata=nodata,
    )

    return data_reprojected




def extract_netcdf_coords_from_rasterio_raster(raster: rasterio.DatasetReader) -> Dict[str, np.array]:
    """Helper to convert rasterio transform in Xarray coordinates.
    """
    transform = raster.transform

    x_scale, x_off, y_scale, y_off = transform.a, transform.c, transform.e, transform.f
    # transform origin is half pixel away from first pixel point
    x0, y0 = x_off + x_scale / 2, y_off + y_scale / 2

    n_cols, n_rows = raster.width, raster.height
    # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
    # Compensate for it
    x_coord = np.arange(n_cols) * x_scale + x0
    y_coord = np.arange(n_rows) * y_scale + y0
    return xr.Coordinates({"y": y_coord, "x": x_coord})
