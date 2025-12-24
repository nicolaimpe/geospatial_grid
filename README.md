# geospatial_grid

## Description

This package allows explicit handling of projections grid for comfortable and automized Xarray objects geographical reprojections.

It provides grid object (GSGrid) to be used together with [rioxarray reproject](https://corteva.github.io/rioxarray/html/rioxarray.html#rioxarray.raster_array.RasterArray.reproject)

## Installation

```bash
git clone git@github.com:nicolaimpe/geospatial_grid.git
cd geospatial_grid
pip install .
```

## Usage
```python
# Define your grid object containing the information for the target projection

from geospatial_grid.gsgrid import GSGrid

my_grid = GSGrid(
            crs=CRS.from_epsg(4326),
            resolution=0.02,
            x0=0,
            y0=40,
            width=10,
            height=10,
            name="GEO_0.01deg",
        )

print("Rasterio representation:")
print(my_grid.affine)
print("Xarray representation:")
print(my_grid.xarray_coords)
```
```
Rasterio representation:
| 0.02, 0.00, 0.00|
| 0.00,-0.02, 40.00|
| 0.00, 0.00, 1.00|
Xarray representation:
Coordinates:
  * y        (y) float64 80B 39.99 39.97 39.95 39.93 ... 39.87 39.85 39.83 39.81
  * x        (x) float64 80B 0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19
```

```python
# Use the grid object for object-oriented regridding
import xarray as xr
from geospatial_grid.reprojections import reproject_using_grid

my_data = xr.open_dataset("my_data.nc")
my_regridded_data = reproject_using_grid(data=my_data, output_grid=my_grid)

```
See `notebooks/example_usage.ipynb` for use cases.

## Contributing

Contributions are welcome.

PDM is recommended for environment management.

```bash
pip install pdm
pdm install
```