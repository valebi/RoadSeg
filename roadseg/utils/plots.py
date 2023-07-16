import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import ImageGrid


def _plot_batch(imgs, masks, preds=None, src="", log_dir=None):
    # @TODO: change to col-first order (cast and reshape)

    if imgs.shape[1] < imgs.shape[2]:
        # channel first -> channel last (assume binary classification)
        imgs = imgs.transpose([0, 2, 3, 1])

    n_samples = len(imgs)
    preds = preds if preds is not None else []
    images = list(imgs) + list(masks) + list(preds)
    labels = (
        [f"{src} Satellite Image"] * len(imgs)
        + [f"{src} Label Mask"] * len(masks)
        + [f"{src} predicted Mask"] * len(preds)
    )

    nrows = 2 if len(preds) == 0 else 3
    fig = plt.figure(figsize=(5 * n_samples, 5 * len(images) // n_samples))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, n_samples),  # creates nrowsx2 grid of axes
        axes_pad=0.5,  # pad between axes in inch.
    )

    for ax, im, lbl in zip(grid, images, labels):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(lbl)

    # fig.draw()
    if log_dir is not None:
        if len(preds) == 0:
            fig.savefig(os.path.join(log_dir, f"sample_batch_{src}.png"))
        else:
            fig.savefig(os.path.join(log_dir, f"sample_pred_{src}.png"))

    # plt.show()
    plt.close("all")


def plot_batch(imgs, msks, pred=None, num_cols=5, src="", log_dir=None):
    if len(msks.shape) == 4:
        msks = msks[:, 0]
    num_cols = min(num_cols, len(imgs))
    num_rows = 2 if pred is None else 3
    fig = plt.figure(figsize=(20, 20 + (pred is None) * 7))
    fig.suptitle(src)
    grid = axes_grid1.ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.1)

    for i, (ax, im, label) in enumerate(zip(grid[:num_cols], imgs[:num_cols], msks[:num_cols])):
        ax.imshow(np.transpose(im, [1, 2, 0]))
        if len(label.shape) == 3:
            grid[i + num_cols].imshow(np.transpose(label, [1, 2, 0]))
        else:
            grid[i + num_cols].imshow(np.expand_dims(label, -1))
        if pred is not None:
            if len(pred[i].shape) == 3:
                grid[i + 2 * num_cols].imshow(np.transpose(pred[i], [1, 2, 0]))
            else:
                grid[i + 2 * num_cols].imshow(np.expand_dims(pred[i], -1))

    if log_dir is not None:
        if pred is None:
            fig.savefig(os.path.join(log_dir, f"sample_batch_{src}.png"))
        else:
            fig.savefig(os.path.join(log_dir, f"sample_pred_{src}.png"))

    # plt.show()
    plt.close("all")
