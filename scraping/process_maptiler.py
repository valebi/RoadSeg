import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from util import config
from util.blur_detector import is_blurred
from util.color_remapping import explore_canvas_colors, encode_images
from util.plots import plot_images


def generate_paths(scale, i, j, processed=False):
    if not processed:
        canvas_path = os.path.join(config.MAPTILER_RENDERING_DIR, "canvas", str(scale), str(i), f"{j}.png")
        image_path = os.path.join(config.MAPTILER_RENDERING_DIR, "sat", str(scale), str(i), f"{j}.png")
    else:
        canvas_path = os.path.join(config.MAPTILER_PROCESSED_DIR, "masks", str(scale), str(i), f"{j}.png")
        image_path = os.path.join(config.MAPTILER_PROCESSED_DIR, "images", str(scale), str(i), f"{j}.jpg")
    return canvas_path, image_path


def process(tile, sanity_check=True, plot=False):
    canvas_dir, img_dir = generate_paths(*tile)
    try:
        sat_image = np.asarray(Image.open(img_dir))[:,:,:3] # Remove alpha from RGBA
        canvas_image = np.asarray(Image.open(canvas_dir))
    except FileNotFoundError:
        return

    # transform
    canvas_image = encode_images(canvas_image, config.MAPTILER_DEFAULT_CENTROIDS, denoise=False)[0]

    if sanity_check:
        num_road_px = np.count_nonzero(canvas_image == 0)
        blurred = is_blurred(sat_image)
        if num_road_px < 100 or blurred:
            return

    if plot:
        plot_images(sat_image, canvas_image)

    # save
    canvas_path, image_path = generate_paths(*tile, processed=True)
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if not os.path.isdir(os.path.dirname(canvas_path)):
        os.makedirs(os.path.dirname(canvas_path))

    Image.fromarray(sat_image).save(image_path)
    Image.fromarray((255 / 5 * canvas_image).astype(np.uint8)).save(canvas_path)


def process_all(subset_size=-1, chunksize=1000):
    """
    processes the ESRI tiles in the directories specified in config
    that is, it picks all double-even tiles and combines it with the odd tiles around them to one 2x2 tile
    then it remaps the colors to class labels and saves the result
    :param subset_size: number of (randomly chosen) images to process
    :param chunksize: chunks to process them in (for progress bar)
    :return:
    """
    files = glob.glob(os.path.join(config.MAPTILER_RENDERING_DIR, "sat", "*", "*", "*.png"))
    np.random.shuffle(files)
    extract_index = lambda fname: list(map(lambda x: int(x.replace(".png", "")), fname.split(os.sep)[-3:]))
    tile_indices = list(map(extract_index, files))
    subset_size = subset_size if subset_size != -1 else len(tile_indices)
    print(f"Beginning to process {len(tile_indices)} tiles.")
    pool = Pool(16)
    for i in tqdm(range(0,min(subset_size,len(tile_indices)), chunksize)):
        pool.map(process, tile_indices[i:i+chunksize])

    out = glob.glob(os.path.join(config.MAPTILER_PROCESSED_DIR, "masks", "*", "*", "*.png"))
    print(f"Total: {len(out)} tile pairs")


if __name__ == "__main__":
    process_all()