from typing import Tuple

import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.transform import from_origin


class GisGridError(Exception):
    pass


class GisGrid:
    def __init__(
        self,
        resolution: float | int | np.float64 | Tuple[float, float],
        x0: float,
        y0: float,
        width: int,
        height: int,
        crs: CRS | None = None,
        name: str | None = None,
    ) -> None:
        self.crs = crs
        if type(resolution) is float or type(resolution) is int or type(resolution) is np.float64:
            self.resolution_x = resolution
            self.resolution_y = resolution
        elif len(resolution) == 2:
            self.resolution_x = resolution[0]
            self.resolution_y = resolution[1]
        else:
            raise GisGridError("Problem with resolution argument")
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.name = name

    """
    Regularly spaced grid
    x0,y0,xend,yend pixel corners
    xmin, ymin, xmax, ymax pixel centers
    width, height, number of pixel columns/rows

    x0,  y0 ------------------------------................--------------xend, y0
        |               |               |                   |               |
        |       .       |       .       | ................  |      .        |
        |   xmin,ymax   |               |                   |   xmax,ymax   |
        |               |               |                   |               |
        |---------------------------------.................------------------
        |               |               |                   |               |
        |               |               |                   |               |
        |

    
    """

    @property
    def xmin(self):
        return self.x0 + self.resolution_x / 2

    @property
    def ymax(self):
        return self.y0 - self.resolution_y / 2

    @property
    def xmax(self):
        return self.xmin + (self.width - 1) * self.resolution_x

    @property
    def ymin(self):
        return self.ymax - (self.height - 1) * self.resolution_y

    @property
    def xend(self):
        return self.x0 + self.width * self.resolution_x

    @property
    def yend(self):
        return self.y0 - self.height * self.resolution_y

    @property
    def extent_llx_lly_urx_ury(self):
        return self.x0, self.yend, self.xend, self.y0

    @property
    def xcoords(self) -> np.array:
        return np.linspace(self.xmin, self.xmax, self.width)

    @property
    def ycoords(self) -> np.array:
        return np.linspace(self.ymax, self.ymin, self.height)

    @property
    def affine(self) -> Affine:
        return from_origin(self.x0, self.y0, self.resolution_x, self.resolution_y)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def xarray_coords(self) -> xr.Coordinates:
        return xr.Coordinates({"y": self.ycoords, "x": self.xcoords})

    def bounds_projected_to_epsg(self, target_epsg: int | str):
        transformer = Transformer.from_crs(crs_from=self.crs, crs_to=CRS.from_epsg(target_epsg), always_xy=True)
        return transformer.transform_bounds(*self.extent_llx_lly_urx_ury)

    @classmethod
    def from_xarray(cls, data: xr.Dataset | xr.DataArray):
        """Be very careful"""

        res_x = data.rio.transform().a
        res_y = data.rio.transform().e

        y_coords, x_coords = data.coords["y"].values, data.coords["x"].values

        width, height = len(x_coords), len(y_coords)

        if not np.array_equal(x_coords, np.arange(x_coords[0], x_coords[0] + width * res_x, res_x)) or not np.array_equal(
            y_coords, np.arange(y_coords[0], y_coords[0] + height * res_y, res_y)
        ):
            raise GisGridError("Data need to be on a reguraly spaced grid")

        if res_y > 0:
            raise GisGridError(
                "Dataset/DatArray y coordinates have to be decreasing (from North to South) to use this function."
            )
        return cls(
            crs=data.rio.crs,
            resolution=(res_x, np.abs(res_y)),
            x0=data.rio.transform().c,
            y0=data.rio.transform().f,
            width=width,
            height=height,
        )
