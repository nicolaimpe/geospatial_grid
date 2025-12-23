import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS
import pandas as pd
from geospatial_grid.reprojections import reproject_using_grid, reproject_onto
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
import pytest


test_array = np.zeros(shape=(3, 10, 10))
test_array[:, :, ::2] = 1
test_data_array = xr.DataArray(
    test_array,
    coords={"x": np.arange(0, 10), "y": np.arange(10, 0, -1), "t": pd.date_range("20251201", "20251203")},
    dims=("t", "y", "x"),
)
test_data_array_georef = georef_netcdf_rioxarray(test_data_array, crs=CRS.from_epsg(3857))

test_data_array_2 = xr.DataArray(
    0,
    coords={"x": np.arange(0, 10), "y": np.arange(10, 0, -1), "t": pd.date_range("20251201", "20251203")},
    dims=("t", "y", "x"),
)
test_data_array_2_georef = georef_netcdf_rioxarray(test_data_array_2, crs=CRS.from_epsg(3857))

test_dataset_georef = xr.Dataset({"tda1": test_data_array_georef, "tda2": test_data_array_2_georef})


@pytest.mark.parametrize(("data"), (test_data_array_georef, test_dataset_georef))
class TestReprojectingUsingGrid:
    test_resampling = Resampling.nearest

    def test_coordinate_system(self, data: xr.DataArray | xr.Dataset):
        test_grid = GSGrid(x0=0, y0=5, resolution=(1, 1), width=5, height=5, crs=CRS.from_epsg(3857))
        test_reprojected = reproject_using_grid(data=data, output_grid=test_grid, resampling_method=self.test_resampling)
        assert test_reprojected.coords["x"][0] == 0.5
        assert test_reprojected.coords["x"][-1] == 4.5
        assert test_reprojected.coords["y"][0] == 4.5
        assert test_reprojected.coords["y"][-1] == 0.5

    def test_crs(self, data: xr.DataArray | xr.Dataset):
        test_crs = CRS.from_epsg(4326)
        test_grid = GSGrid(x0=0, y0=5, resolution=(1, 1), width=5, height=5, crs=test_crs)
        test_reprojected = reproject_using_grid(data=data, output_grid=test_grid, resampling_method=self.test_resampling)
        reprojected_crs = CRS.from_wkt(test_reprojected.coords["spatial_ref"].attrs["spatial_ref"])
        assert reprojected_crs.name == test_crs.name
        for i in range(1):
            assert reprojected_crs.axis_info[i].name == test_crs.axis_info[i].name
            assert reprojected_crs.axis_info[i].unit_code == test_crs.axis_info[i].unit_code

    def test_change_resolution(self, data: xr.DataArray | xr.Dataset):
        test_grid = GSGrid(x0=0, y0=10, resolution=(2, 0.5), width=5, height=3, crs=CRS.from_epsg(3857))
        test_reprojected = reproject_using_grid(data=data, output_grid=test_grid, resampling_method=self.test_resampling)
        assert test_reprojected.coords["x"][0] == 1
        assert test_reprojected.coords["x"][-1] == 9
        assert test_reprojected.coords["y"][0] == 9.75
        assert test_reprojected.coords["y"][-1] == 8.75


def test_reproject_using_grid_bilinear_resampling():
    test_grid = GSGrid(x0=0, y0=10, resolution=(1, 1), width=5, height=5, crs=CRS.from_epsg(3857))
    test_reprojected = reproject_using_grid(
        data=test_data_array_georef, output_grid=test_grid, resampling_method=Resampling.bilinear
    )
    assert np.array_equal(test_reprojected.values, 0.5 * np.ones_like(test_reprojected.values))


def test_reproject_using_grid_nearest_resampling():
    test_grid = GSGrid(x0=0.1, y0=10, resolution=(1, 1), width=5, height=5, crs=CRS.from_epsg(3857))
    test_reprojected = reproject_using_grid(
        data=test_data_array_georef, output_grid=test_grid, resampling_method=Resampling.nearest
    )
    expected_reprojected_array = np.ones(shape=(3, 5, 5))
    expected_reprojected_array[:, :, ::2] = 0
    assert np.array_equal(test_reprojected.values, expected_reprojected_array)


def test_reproject_dataset():
    test_grid = GSGrid(x0=0, y0=10, resolution=(1, 1), width=5, height=5, crs=CRS.from_epsg(3857))
    test_reprojected = reproject_using_grid(
        data=test_dataset_georef, output_grid=test_grid, resampling_method=Resampling.bilinear
    )
    assert np.array_equal(
        test_reprojected.data_vars["tda1"].values, 0.5 * np.ones_like(test_reprojected.data_vars["tda1"].values)
    )
    assert np.array_equal(test_reprojected.data_vars["tda2"].values, np.zeros_like(test_reprojected.data_vars["tda2"].values))


def test_reproject_onto():
    target_data_array = xr.DataArray(
        1,
        coords={"x": np.arange(0, 10e4, 1e3), "y": np.arange(10e4, 2000, -1e3)},
        dims=("y", "x"),
    )
    target_data_array_georef = georef_netcdf_rioxarray(data_array=target_data_array, crs=CRS.from_epsg(3395))
    test_data_array_1 = xr.DataArray(
        1,
        coords={
            "x": np.arange(-10e5, 10e5, 1e4),
            "y": np.arange(10e5, -10e5, -1e4),
            "t": pd.date_range("20251201", "20251203"),
        },
        dims=("t", "y", "x"),
    )
    test_data_array_2 = xr.DataArray(
        2,
        coords={
            "x": np.arange(-10e5, 10e5, 1e4),
            "y": np.arange(10e5, -10e5, -1e4),
            "t": pd.date_range("20251201", "20251203"),
        },
        dims=("t", "y", "x"),
    )
    target_data_array_1_georef = georef_netcdf_rioxarray(data_array=test_data_array_1, crs=CRS.from_epsg(3857))
    target_data_array_2_georef = georef_netcdf_rioxarray(data_array=test_data_array_2, crs=CRS.from_epsg(3857))
    test_dataset_georef = xr.Dataset({"tda1": target_data_array_1_georef, "tda2": target_data_array_2_georef})
    reprojected_dataset = reproject_onto(
        test_dataset_georef, target_data_array_georef, resampling_method=Resampling.nearest, nodata=255
    )
    # Test that the operation is done for each time coordinate of the time series
    for time in test_data_array_1.coords["t"].values:
        # We need to drop time because it's not on the target data array and spatial_ref because it's a coordinate of the Dataset object
        # and hence not when we extract a time coordinate
        reprojected_t = reprojected_dataset.sel(t=time).data_vars["tda1"]
        reprojected_t = reprojected_t.drop_vars(("t", "spatial_ref"))
        assert target_data_array.equals(reprojected_t)
    for time in test_data_array_2.coords["t"].values:
        # We need to drop time because it's not on the target data array and spatial_ref because it's a coordinate of the Dataset object
        # and hence not when we extract a time coordinate
        reprojected_t = reprojected_dataset.sel(t=time).data_vars["tda2"]
        reprojected_t = reprojected_t.drop_vars(("t", "spatial_ref"))
        assert reprojected_t.equals(2 * target_data_array)
