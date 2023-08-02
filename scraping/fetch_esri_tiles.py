import os.path
import urllib.request

from tqdm import tqdm
import time

from util.cities import get_cities, coords2tile_OSM
from util.config import ESRI_DATA_DIR


def fetch_road_tile(z,y,x, overwrite=False):
    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}.jpg"
    dir = f"{ESRI_DATA_DIR}/streetmap/{z}/{y}/{x}.jpg"
    try:
        os.makedirs(os.path.dirname(dir))
    except:
        pass
    if overwrite or not os.path.isfile(dir):
        urllib.request.urlretrieve(url, dir)


def fetch_sat_tile(z,y,x, overwrite=False):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg"
    dir = f"{ESRI_DATA_DIR}/sat/{z}/{y}/{x}.jpg"
    try:
        os.makedirs(os.path.dirname(dir))
    except:
        pass
    if overwrite or not os.path.isfile(dir):
        urllib.request.urlretrieve(url, dir)


if __name__ == "__main__":
    tiles_per_city = 50
    cities = get_cities(min_population=100000)
    spec = []
    for i, (name, (long, lat)) in enumerate(cities):
        print(f"Fetching data for {name} [{i+1}/{len(cities)}]")
        s, y, x = coords2tile_OSM(long, lat, 18)
        tiles_around = [(s, y + i, x + j) for i in range(-tiles_per_city // 2, tiles_per_city // 2) for j in range(-tiles_per_city // 2, tiles_per_city // 2)]
        for tile in tqdm(tiles_around):
            try:
                fetch_road_tile(*tile)
            except:
                print("Failed to fetch road tile", tile)
                time.sleep(2)
            try:
                fetch_sat_tile(*tile)
            except:
                print("Failed to fetch sat tile", tile)
                time.sleep(2)
