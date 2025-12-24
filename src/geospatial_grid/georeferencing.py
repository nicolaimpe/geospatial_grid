import xarray as xr
import pyproj
import rioxarray


def georef_netcdf_manually(data_array: xr.DataArray | xr.Dataset, crs: pyproj.CRS) -> xr.Dataset | xr.Dataset:
    """
    The strict minimum to georeference in netCDF convention

    Enforce georeferencing for GDAL/QGIS
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    # dims = dim_name(crs=crs)
    data_array.coords["y"].attrs["axis"] = "Y"
    data_array.coords["x"].attrs["axis"] = "X"
    data_array.attrs["grid_mapping"] = "spatial_ref"

    georeferenced = data_array.assign_coords(coords={"spatial_ref": 0})
    georeferenced.coords["spatial_ref"].attrs["spatial_ref"] = crs.to_wkt()

    return georeferenced


def georef_netcdf_rioxarray(data_array: xr.DataArray | xr.Dataset, crs: pyproj.CRS) -> xr.Dataset | xr.DataArray:
    """
    Turn a DataArray into a Dataset for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing using rioxarray functions.

    We too often forget write_coordinate_system()
    """

    return data_array.rio.write_crs(crs).rio.write_coordinate_system()


def extract_crs(data: xr.DataArray | xr.Dataset) -> pyproj.CRS:
    """Wrap up rioxarray crs so that it's typed"""
    return data.rio.crs
