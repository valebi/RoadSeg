import os
import time
import urllib

import numpy as np
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from util.cities import coords2tile_OSM, get_cities, tile2coords_OSM
from util.config import MAPTILER_API_KEY, MAPTILER_RENDERING_DIR, ESRI_DATA_DIR, MAPTILER_CANVAS_ID

CONTROL_CLASS_CANVAS = 'maplibregl-control-container'
CONTROL_CLASS_TILES = 'ol-overlaycontainer-stopevent'
CANVAS_CLASS = 'maplibregl-canvas'

def delete_control_elements(driver, is_canvas=True):
    if is_canvas:
        element = driver.find_element_by_class_name(CONTROL_CLASS_CANVAS)
    else:
        element = driver.find_element_by_class_name(CONTROL_CLASS_TILES)
    driver.execute_script("""
    var element = arguments[0];
    element.parentNode.removeChild(element);
    """, element)


def capture_canvas(driver, z, y, x, overwrite=False):
    #driver.get(f'https://api.maptiler.com/maps/hybrid/?key={API_KEY}#{z}/{x}/{y}')
    lat, lon = tile2coords_OSM(z+1,y,x)
    url = f'https://api.maptiler.com/maps/{MAPTILER_CANVAS_ID}/?key={MAPTILER_API_KEY}#{z}/{lat}/{lon}'
    dir = f"{MAPTILER_RENDERING_DIR}/canvas/{z+1}/{y}/{x}.png"
    if overwrite or not os.path.isfile(dir):
        driver.get(url)
        delay = 3  # seconds (max)
        try:
            myElem = WebDriverWait(driver, delay).until(
                EC.frame_to_be_available_and_switch_to_it((By.CLASS_NAME, CANVAS_CLASS)))
        except TimeoutException:
            "Loading took too much time!"
        try:
            delete_control_elements(driver, is_canvas=True)
        except NoSuchElementException:
            pass # we apparently only have to remove it once
        try:
            os.makedirs(os.path.dirname(dir))
        except:
            pass
        driver.save_screenshot(dir)


def capture_sat(driver, z, y, x, overwrite=False):
    #driver.get(f'https://api.maptiler.com/maps/hybrid/?key={API_KEY}#{z}/{x}/{y}')
    lat, lon = tile2coords_OSM(z+1,y,x)
    url = f'https://api.maptiler.com/tiles/satellite-v2/?key={MAPTILER_API_KEY}#{z}/{lat}/{lon}'
    dir = f"{MAPTILER_RENDERING_DIR}/sat/{z+1}/{y}/{x}.png"
    if overwrite or not os.path.isfile(dir):
        driver.get(url)
        delay = 3  # seconds (max)
        try:
            myElem = WebDriverWait(driver, delay).until(
                EC.frame_to_be_available_and_switch_to_it((By.CLASS_NAME, CANVAS_CLASS)))
        except TimeoutException:
            "Loading took too much time!"
        try:
            delete_control_elements(driver, is_canvas=False)
        except NoSuchElementException:
            pass  # we apparently only have to remove it once
        try:
            os.makedirs(os.path.dirname(dir))
        except:
            pass
        driver.save_screenshot(dir)


def generate_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=600,600") # 400,400
    return webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)

if __name__ == "__main__":
    tiles_per_city = 15
    zoom = 17
    cities = get_cities(min_population=100000)
    spec = []
    for i, (name, (long, lat)) in enumerate(cities):
        print(f"Fetching data for {name} [{i+1}/{len(cities)}]")
        driver = generate_driver()
        # calculate center tiles based on OSM grid (to align filenames roughly)
        s, y, x = coords2tile_OSM(long, lat, zoom+1)
        tiles_around = [(y + i, x + j) for i in range(-tiles_per_city, tiles_per_city, 2) for j in range(-tiles_per_city, tiles_per_city, 2)]
        for tile in tqdm(tiles_around):
            try:
                capture_sat(driver, zoom, *tile)
            except:
                print("Failed to fetch sat tile", tile)
                time.sleep(2)
                driver = generate_driver()
            try:
                capture_canvas(driver, zoom, *tile)
            except:
                print("Failed to fetch canvas tile", tile)
                time.sleep(2)
                driver = generate_driver()