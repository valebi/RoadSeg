import glob
import os

import numpy as np
import scipy
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

from util import config
from util.plots import plot_centroids


def compute_centroids(canvas_images, n_clusters):
    print(f"Running kmeans to cluster centroids ({n_clusters} centers)")
    N, H, W, C = canvas_images.shape
    flat = canvas_images.reshape((-1, C)).astype(np.float64)
    codes = KMeans(n_clusters=n_clusters, n_init="auto").fit(flat).cluster_centers_
    indexs = np.argsort(-np.linalg.norm(codes, axis=1))
    codes = codes[indexs]
    vecs, dist = scipy.cluster.vq.vq(flat, codes)         # assign codes
    error = np.sqrt(np.mean(dist))
    coded_images = vecs.reshape((-1,H,W))
    return error, codes, coded_images


def encode_images(images, centroids, denoise=False):
    """
    maps canvas images from RGB to color space
    :param images: (N, h, w, 3) input canvas images
    :param centroids: (n_clusters, 3) the cluster centers in color space corresponding to semantic classes
    :param denoise:
    :return:
    """
    h, w, c = images.shape[-3:]
    vecs, dist = scipy.cluster.vq.vq(images.reshape((-1, c)), centroids)  # distance transform
    clean_dist, coded_images = dist.reshape((-1, h, w)), vecs.reshape((-1, h, w))
    if denoise:
        coded_images = np.asarray([scipy.ndimage.median_filter(img.copy(), 3) for img in coded_images])
    return coded_images


def relabel_coded_image(coded_images, mapping):
    _coded_images = np.copy(coded_images)
    for alias in config.ESRI_CENTROID_REMAPPING.keys():
        coded_images[_coded_images == alias] = mapping[alias]
    return coded_images


def explore_canvas_colors(run_different_nclusters=False, plot_config=True, n_files=1000, source="ESRI"):
    if source == "ESRI":
        files = np.array(glob.glob(os.path.join(config.ESRI_DATA_DIR, "streetmap", "*", "*", "*.jpg")))
        default_centroids = config.ESRI_DEFAULT_CENTROIDS
    else:
        # MAPTILER
        files = np.array(glob.glob(os.path.join(config.MAPTILER_RENDERING_DIR, "canvas", "*", "*", "*.png")))
        default_centroids = config.MAPTILER_DEFAULT_CENTROIDS

    print(f"Found {len(files)} files")
    np.random.shuffle(files)
    files = files[:n_files]
    images = np.stack([np.array(Image.open(i)) for i in tqdm(files)])
    if run_different_nclusters:
        errors = []
        for nc in range(4, 10):
            error, codes, coded_images = compute_centroids(images, nc)
            plt.title(f"n_centroids={nc} RMSE={error}")
            plot_centroids(codes, coded_images)
            errors.append(error)

        plt.title("Error as function of centroids")
        plt.plot(errors)
        plt.show()

    if plot_config:
        codes = np.array(default_centroids)
        coded_images = encode_images(images, codes)
        if source == "ESRI":
            coded_images = relabel_coded_image(coded_images, config.ESRI_CENTROID_REMAPPING)
        plot_centroids(codes, coded_images)
        print(codes)
        for c in range(min(len(codes), len(np.unique(coded_images.flatten())))):
            if np.count_nonzero(c == coded_images) > 0:
                plt.figure(figsize=(30, 30))
                for i in range(16):
                    plt.subplot(4,4,i+1)
                    plt.title(f"{c} Class")
                    plt.imshow((c == coded_images)[i])
                plt.show()
                plt.figure(figsize=(30, 30))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(images[i])
        plt.show()
