import numpy as np
from matplotlib import pyplot as plt


def plot_centroids(codes, coded_images):
    plt.subplot(1,3,1)
    for i in range(len(codes)):
        plt.plot((0, 1), ((i+1) / 12, (i+1) / 6), lw=8, color=tuple(codes[i] / 255))
        print(tuple(codes[i] / 255))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"{len(codes)} centers")
    plt.gca().set_aspect('equal')
    plt.subplot(1, 3, 2)
    ix = np.random.randint(0, len(coded_images))
    plt.imshow(coded_images[ix])
    plt.gca().set_aspect('equal')
    plt.subplot(1, 3, 3)
    plt.imshow(coded_images[ix] == 0)
    plt.show()


def plot_images(sat_image, canvas_image):
    plt.subplot(1, 2, 1)
    plt.imshow(sat_image)
    plt.subplot(1, 2, 2)
    plt.imshow(canvas_image)
    plt.show()