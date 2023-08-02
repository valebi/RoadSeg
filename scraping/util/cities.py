import math

import pandas as pd

from util.config import GEONAMES_FILE


def get_cities(country_code="US", min_population=500000):
    cities = pd.read_csv(GEONAMES_FILE, sep=";",
                on_bad_lines='skip', header=0)
    if min_population is not None:
        cities = cities[cities["Population"] > min_population]
    if country_code is not None:
        cities = cities[cities["Country Code"] == country_code]
    to_float = lambda x: tuple(map(float, x.split(", ")))
    return list(zip(cities["Name"], map(to_float, cities["Coordinates"])))


def coords2tile_OSM(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (zoom, ytile, xtile)



def tile2coords_OSM(zoom, ytile, xtile):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)
