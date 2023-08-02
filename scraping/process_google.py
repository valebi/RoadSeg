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


def process(img_dir, sanity_check=True, plot=False):
    road_dir = img_dir.replace("sat", "road")
    try:
        sat_image = np.asarray(Image.open(img_dir))[:,:,:3] # Remove alpha from RGBA
        _road_image = np.asarray(Image.open(road_dir))
    except Exception as e:
        print(e)
        return

    road_image = _road_image > 50
    if sanity_check and not "extra" in road_dir:
        num_road_px = np.count_nonzero(road_image)
        blurred = is_blurred(sat_image)
        min_px = sat_image.shape[0]*sat_image.shape[1]*0.01
        if num_road_px < min_px or blurred:
            # print(num_road_px,blurred, np.max(_road_image))
            return

    if plot:
        plot_images(sat_image, road_image)

    # save
    image_path, canvas_path = img_dir.replace(config.GOOGLE_DATA_DIR, config.GOOGLE_PROCESSED_DIR), road_dir.replace(config.GOOGLE_DATA_DIR, config.GOOGLE_PROCESSED_DIR)
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if not os.path.isdir(os.path.dirname(canvas_path)):
        os.makedirs(os.path.dirname(canvas_path))

    Image.fromarray(sat_image).save(image_path)
    Image.fromarray((255 * road_image).astype(np.uint8)).save(canvas_path)


def process_all(subset_size=-1, chunksize=1000):
    """
    processes the ESRI tiles in the directories specified in config
    that is, it picks all double-even tiles and combines it with the odd tiles around them to one 2x2 tile
    then it remaps the colors to class labels and saves the result
    :param subset_size: number of (randomly chosen) images to process
    :param chunksize: chunks to process them in (for progress bar)
    :return:
    """
    files = glob.glob(os.path.join(config.GOOGLE_DATA_DIR, "sat", "*", "*.png"))
    np.random.shuffle(files)
    print(f"Beginning to process {len(files)} tiles.")
    pool = Pool(16)
    # for i in tqdm(range(0,min(subset_size,len(files)), chunksize)):
    #     pool.map(process, files[i:i+chunksize])
    for f in tqdm(files):
        process(f)
    out = glob.glob(os.path.join(config.GOOGLE_PROCESSED_DIR, "sat", "*", "*.png"))
    print(f"Total: {len(out)} tile pairs")


if __name__ == "__main__":
    process_all()