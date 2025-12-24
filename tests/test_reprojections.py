import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS
import pandas as pd
from geospatial_grid.reprojections import reproject_using_grid
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

