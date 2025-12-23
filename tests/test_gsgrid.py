import numpy as np
from geospatial_grid.gsgrid import GSGrid, GSGridError
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
import pytest
import xarray as xr
from pyproj import CRS


def test_resolution_argument():
    with pytest.raises(GSGridError):
        GSGrid(resolution=(1, 1, 1), x0=0, y0=1, width=100, height=100)


def test_grid_bounds():
    test_grid = GSGrid(resolution=(1, 2), x0=0, y0=1, width=200, height=100)
    assert test_grid.xend == test_grid.xmax + 0.5
    assert test_grid.yend == test_grid.ymin - 1
    assert test_grid.xcoords[0] == test_grid.extent_llx_lly_urx_ury[0] + 0.5
    assert test_grid.xcoords[-1] == test_grid.extent_llx_lly_urx_ury[2] - 0.5
    assert test_grid.ycoords[0] == test_grid.extent_llx_lly_urx_ury[3] - 1
    assert test_grid.ycoords[-1] == test_grid.extent_llx_lly_urx_ury[1] + 1


def test_grid_affine():
    test_grid = GSGrid(resolution=(1, 2), x0=0, y0=1, width=200, height=100)
    assert test_grid.affine.a == 1
    assert test_grid.affine.b == 0
    assert test_grid.affine.c == 0
    assert test_grid.affine.d == 0
    assert test_grid.affine.e == -2
    assert test_grid.affine.f == 1


def test_grid_xarray_coords():
    test_grid = GSGrid(resolution=(1, 2), x0=0, y0=0, width=200, height=100)
    assert test_grid.xarray_coords["x"][100] == 100.5
    assert test_grid.xarray_coords["y"][49] == -99


def test_grid_from_dataset():
    # Not evenly spaced on x
    test_bad_data_array = georef_netcdf_rioxarray(
        xr.DataArray(0, coords={"x": np.array([0, 5, 8]), "y": np.arange(0, 201, 2)}, dims=("x", "y")),
        crs=CRS.from_epsg(4326),
    )
    with pytest.raises(GSGridError):
        GSGrid.from_xarray(data=test_bad_data_array)
    # Not evenly spaced on y
    test_bad_data_array = georef_netcdf_rioxarray(
        xr.DataArray(0, coords={"x": np.arange(0, 201, 1), "y": np.array([0, 5, 8])}, dims=("x", "y")),
        crs=CRS.from_epsg(4326),
    )
    with pytest.raises(GSGridError):
        GSGrid.from_xarray(data=test_bad_data_array)
    # Not North Y-axis
    test_bad_data_array = georef_netcdf_rioxarray(
        xr.DataArray(0, coords={"x": np.arange(0, 201, 1), "y": np.arange(0, 201, 2)}, dims=("x", "y")),
        crs=CRS.from_epsg(4326),
    )
    with pytest.raises(GSGridError):
        GSGrid.from_xarray(data=test_bad_data_array)

    test_data_array = georef_netcdf_rioxarray(
        xr.DataArray(0, coords={"x": np.arange(0, 201, 1), "y": np.arange(200, -1, -2)}, dims=("x", "y")),
        crs=CRS.from_epsg(4326),
    )
    test_grid = GSGrid.from_xarray(data=test_data_array)
    assert test_grid.x0 == -0.5
    assert test_grid.y0 == 201
    assert test_grid.xend == 200.5
    assert test_grid.yend == -1
    assert test_grid.resolution_x == 1
    assert test_grid.resolution_y == 2
    assert test_grid.xmin == 0
    assert test_grid.ymin == 0
    assert test_grid.xmax == 200
    assert test_grid.ymax == 200
