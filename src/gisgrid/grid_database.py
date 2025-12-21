from gisgird.gisgrid import GisGrid
from pyproj import CRS


# MODIS SIN grid
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"


class UTM375mGrid(GisGrid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_epsg(32631),
            resolution=375,
            x0=0,
            y0=5400000,
            width=2800,
            height=2200,
            name="UTM_375m",
        )


class SIN375mGrid(GisGrid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_proj4(PROJ4_MODIS),
            resolution=370.650173222222,
            x0=-420000,
            y0=5450000,
            width=3500,
            height=2600,
            name="SIN_375m",
        )


class LatLon375mGrid(GisGrid):
    def __init__(self):
        super().__init__(
            crs=CRS.from_epsg(4326),
            resolution=(0.003374578177758, 0.0033740359897170007),
            x0=-5.0033746,
            y0=51.496626,
            width=4447,
            height=3112,
            name="GEO_375m",
        )
