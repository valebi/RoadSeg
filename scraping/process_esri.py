import glob
import os
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from util import config
from util.blur_detector import is_blurred
from util.color_remapping import encode_images, relabel_coded_image, explore_canvas_colors
from util.plots import plot_centroids, plot_images




def assemble(images):
    vert = np.concatenate(images[:2], axis=0)
    horz = np.concatenate(images[2:], axis=0)
    return np.concatenate([vert, horz], axis=1)


def generate_paths(scale, i, j, processed=False):
    if not processed:
        canvas_path = os.path.join(config.ESRI_DATA_DIR, "streetmap", str(scale), str(i), f"{j}.jpg")
        image_path = os.path.join(config.ESRI_DATA_DIR, "sat", str(scale), str(i), f"{j}.jpg")
    else:
        canvas_path = os.path.join(config.ESRI_PROCESSED_DIR, "masks", str(scale), str(i), f"{j}.png")
        image_path = os.path.join(config.ESRI_PROCESSED_DIR, "images", str(scale), str(i), f"{j}.png")
    return canvas_path, image_path


def process_tile(tile, sanity_check=True, plot=False):
    """
    processes a tile. Combines it with the surrounding tiles and maps the canvas to class masks (0-4)
    fails silently if so.
    :param tile: tuple (scale, i, j) = tile descriptor
    :param sanity_check: whether to check for blurred tiles
    :param plot: shows result if true
    :return:
    """
    # load surrounding tiles
    d, u, v = tile
    tiles_around = [(d, u, v), (d, u+1, v),(d, u, v+1),(d, u+1, v+1)]
    canvas_paths, image_paths = zip(*[generate_paths(*t) for t in tiles_around])
    try:
        sat_images = list(map(np.asarray, map(Image.open, image_paths)))
        canvas_images = list(map(np.asarray, map(Image.open, canvas_paths)))
    except (FileNotFoundError, OSError):
        # files do not exist or corrupted
        return
    # process them as one
    sat_image = assemble(sat_images)
    canvas_image = assemble(canvas_images)
    canvas_image = encode_images(canvas_image, config.ESRI_DEFAULT_CENTROIDS, denoise=True)[0]
    canvas_image = relabel_coded_image(canvas_image, config.ESRI_CENTROID_REMAPPING)

    if plot:
        plot_images(sat_image, canvas_image)

    # check whether they contain something valuable for training
    if sanity_check:
        num_road_px = np.count_nonzero(canvas_image == 0)
        blurred = is_blurred(sat_image)
        if num_road_px < 100 or blurred:
            return

    # save
    canvas_path, image_path = generate_paths(d,u,v, processed=True)
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if not os.path.isdir(os.path.dirname(canvas_path)):
        os.makedirs(os.path.dirname(canvas_path))

    Image.fromarray(sat_image).save(image_path)
    Image.fromarray((255 / 5 * canvas_image).astype(np.uint8)).save(canvas_path)


def process_all(subset_size=-1, chunksize=10000):
    """
    processes the ESRI tiles in the directories specified in config
    that is, it picks all double-even tiles and combines it with the odd tiles around them to one 2x2 tile
    then it remaps the colors to class labels and saves the result
    :param subset_size: number of (randomly chosen) images to process
    :param chunksize: chunks to process them in (for progress bar)
    :return:
    """
    files = glob.glob(os.path.join(config.ESRI_DATA_DIR, "streetmap", "*", "*", "*.jpg"))
    np.random.shuffle(files)
    extract_index = lambda fname: list(map(lambda x: int(x.replace(".jpg", "")), fname.split(os.sep)[-3:]))
    tile_indices = list(map(extract_index, files))
    tile_indices = [t for t in tile_indices if t[1] % 2 == 0 and t[2] % 2 == 0]
    subset_size = subset_size if subset_size != -1 else len(files)
    print(f"Beginning to process {len(tile_indices)} tiles.")
    pool = Pool(16)
    for i in tqdm(range(0,min(subset_size,len(tile_indices)), chunksize)):
        pool.map(process_tile, tile_indices[i:i+chunksize])

    out = glob.glob(os.path.join(config.ESRI_PROCESSED_DIR, "masks", "*", "*", "*.png"))
    print(f"Total: {len(out)} tile pairs")


if __name__ == "__main__":
    explore_canvas_colors()
    #process_all()
    with Image.open("esri_tiles_processed/masks/18/105262/49648.png") as img:
        plt.imshow(img)
        plt.show()
        print(np.asarray(img))