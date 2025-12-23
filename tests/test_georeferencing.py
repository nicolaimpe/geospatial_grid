import numpy as np
import xarray as xr
import rioxarray
from pyproj import CRS
from gisgrid.georeferencing import georef_netcdf_manually, georef_netcdf_rioxarray
import pandas as pd
import pytest

test_data_array = xr.DataArray(0, coords={"x": np.arange(20, 30), "y": np.arange(10, -30, -1)}, dims=("x", "y"))
test_data_array_georef_rioxarray = georef_netcdf_rioxarray(test_data_array, crs=CRS.from_epsg(4326))
test_data_array_georef_manually = georef_netcdf_manually(test_data_array, crs=CRS.from_epsg(4326))

test_data_array_1 = xr.DataArray(
    0,
    coords={"x": np.arange(20, 30), "y": np.arange(10, -30, -1), "t": pd.date_range("20251201", "20251231")},
    dims=("t", "y", "x"),
)
test_data_array_2 = xr.DataArray(
    1,
    coords={"x": np.arange(20, 30), "y": np.arange(10, -30, -1), "t": pd.date_range("20251201", "20251231")},
    dims=("t", "y", "x"),
)
test_dataset = xr.Dataset({"tda1": test_data_array_1, "tda2": test_data_array_2})
test_dataset_georef_rioxarray = georef_netcdf_rioxarray(test_dataset, crs=CRS.from_epsg(4326))
test_dataset_georef_manually = georef_netcdf_manually(test_dataset, crs=CRS.from_epsg(4326))


@pytest.mark.parametrize(
    ("data_georef_rioxarray", "data_georef_manually"),
    (
        [test_data_array_georef_manually, test_data_array_georef_rioxarray],
        [test_dataset_georef_manually, test_dataset_georef_rioxarray],
    ),
)
def test_mutual_georef_data_array(
    data_georef_manually: xr.DataArray | xr.Dataset, data_georef_rioxarray: xr.DataArray | xr.Dataset
):
    crs_rioxarray = CRS.from_wkt(data_georef_rioxarray.coords["spatial_ref"].attrs["spatial_ref"])
    crs_manual = CRS.from_wkt(data_georef_manually.coords["spatial_ref"].attrs["spatial_ref"])
    assert crs_manual.name == crs_rioxarray.name
    assert crs_manual.coordinate_system == crs_rioxarray.coordinate_system
    assert test_data_array_georef_manually.coords["x"].attrs["axis"] == "X"
    assert test_data_array_georef_rioxarray.coords["x"].attrs["axis"] == "X"
    assert test_data_array_georef_manually.coords["y"].attrs["axis"] == "Y"
    assert test_data_array_georef_rioxarray.coords["y"].attrs["axis"] == "Y"
