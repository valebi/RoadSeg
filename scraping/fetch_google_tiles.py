""" built with help / using code blocks from Yanheng Wang, Tianfu Wang, Noel Boos and Pascal Trocker (and their 2022 CIL course project) """
import json
import math
import os
import time
import urllib.request

import googlemaps
import requests
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
from shapely import wkt
import csv
import random
import pandas as pd
from sklearn.cluster import KMeans
import numpy
from tqdm import tqdm

from util.cities import coords2tile_OSM, get_cities
from util.config import GOOGLE_DATA_DIR, GOOGLE_API_KEY

numpy.random.seed(seed=0)

api_key = GOOGLE_API_KEY
prefix = "https://maps.googleapis.com/maps/api/staticmap?key="+api_key+"&center="

suffix_sat = "&format=png32&maptype=satellite&size="
suffix_road = "&format=png&maptype=roadmap&style=visibility:off&style=element:geometry.fill%7Ccolor:0x000000%7Cvisibility:on&style=feature:road%7Cvisibility:on&style=feature:road%7Celement:geometry.fill%7Ccolor:0xffffff%7Cvisibility:on&style=feature:road%7Celement:labels%7Cvisibility:off&style=feature:road.highway%7Celement:geometry.fill%7Ccolor:0xffffff&style=feature:road.highway%7Celement:geometry.stroke%7Cvisibility:off&size="
size_w = 600
size_h = 630

final_size = str(size_w)+"x"+str(size_h)

def get_coord_string(tile):
	z,x,y = tile
	coords = str(x) + "," + str(y) + "&zoom=" + str(z)
	return coords



def fetch_road_tile(name, z, y, x, overwrite=False):
	coords = get_coord_string((z,y,x))
	url = prefix + coords + suffix_road + final_size
	dir = f"{GOOGLE_DATA_DIR}/road/{name}/{z}_{y}_{x}.png"
	try:
		os.makedirs(os.path.dirname(dir))
	except:
		pass
	if overwrite or not os.path.isfile(dir):
		urllib.request.urlretrieve(url, dir)
		img = Image.open(dir)
		img = img.crop((0,0,600,600))
		img.save(dir)


def fetch_sat_tile(name, z, y, x, overwrite=False):
	coords = get_coord_string((z,y,x))
	url = prefix + coords + suffix_sat + final_size
	dir = f"{GOOGLE_DATA_DIR}/sat/{name}/{z}_{y}_{x}.png"
	try:
		os.makedirs(os.path.dirname(dir))
	except:
		pass
	if overwrite or not os.path.isfile(dir):
		urllib.request.urlretrieve(url, dir)
		img = Image.open(dir)
		img = img.crop((0,0,600,600))
		img.save(dir)


def offset(tile, long_meters, lat_meters):
	z, long, lat = tile
	offset_lat = 1 / 111111 * lat_meters
	offset_long = 1 / (111111 * math.cos(lat)) * long_meters
	return (z, long + offset_long, lat + offset_lat)


if __name__ == "__main__":
	tiles_per_city = 14
	max_cities = 20
	cities = []
	with open("cities.json") as f:
		ls = json.load(f)
		for c in ls:
			ppl = int(c["population"])
			if ppl > 0.5 * 1000000 and len(cities) < max_cities:
				cities.append((c["city"], (c["longitude"], c["latitude"])))

	print(cities[0])
	cities = [("extra1", (-118.450419, 34.037052)), ("extra2", (-71.112710, 42.334054))]
	print(cities)
	print(f"scraping {tiles_per_city*tiles_per_city*len(cities)*2} images")
	for i, (name, (long, lat)) in enumerate(cities):
		print(f"Fetching data for {name} [{i+1}/{len(cities)}]")
		tile = (18, lat, long)
		# tiles_around = [offset(tile, i*350,j*350) for i in range(-tiles_per_city // 2, tiles_per_city // 2) for j in range(-tiles_per_city // 2, tiles_per_city // 2)]
		tiles_around = [offset(tile, i*225,j*225) for i in range(-tiles_per_city // 2, tiles_per_city // 2) for j in range(-tiles_per_city // 2, tiles_per_city // 2)]
		for tile in tqdm(tiles_around):
			try:
				fetch_road_tile(name, *tile)
			except:
				print("Failed to fetch road tile", tile)
				time.sleep(2)
			try:
				fetch_sat_tile(name, *tile)
			except:
				print("Failed to fetch sat tile", tile)
				time.sleep(2)

