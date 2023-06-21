import glob
import os

import numpy as np
import PIL
import torch
from PIL import Image

from roadseg.utils.mask_to_submission import (
    mask_to_submission_strings,
    masks_to_submission,
    save_mask_as_img,
)


@torch.no_grad()
def generate_predictions(model, CFG, road_class=1, fold=""):
    img_files = [f for f in os.listdir(CFG.test_imgs_dir) if f.endswith(".png")]
    imgs = [np.array(Image.open(os.path.join(CFG.test_imgs_dir, f)))[:, :, :3] for f in img_files]

    model.to(CFG.device)
    model.eval()

    dirname = os.path.join(CFG.out_dir, f"fold-{fold}")
    os.makedirs(dirname, exist_ok=True)
    imgs = np.asarray(imgs).transpose([0, 3, 1, 2]).astype(np.float32)
    imgs /= 255.0

    pred = model(torch.tensor(imgs).to(CFG.device))
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = pred.numpy()[:, road_class, :, :] * 255
    pred = pred.astype(np.uint8)
    for i, prd in enumerate(pred):
        img = PIL.Image.fromarray(prd)
        img.save(os.path.join(dirname, img_files[i]))


def make_ensemble(CFG):
    img_files = [f for f in os.listdir(CFG.test_imgs_dir) if f.endswith(".png")]
    dirname = os.path.join(CFG.out_dir, f"ensemble")
    os.makedirs(dirname, exist_ok=True)
    for i in range(len(img_files)):
        imgs = glob.glob(f"{CFG.out_dir}/fold-*/{img_files[i]}")
        imgs = [np.array(Image.open(img)) for img in imgs]
        ensemble = np.mean(imgs, axis=0).astype(np.uint8)
        PIL.Image.fromarray(ensemble).save(os.path.join(dirname, img_files[i]))


def make_submission(CFG):
    image_filenames = sorted(glob.glob(f"{CFG.out_dir}/ensemble/*.png"))
    masks_to_submission("submission.csv", "", *image_filenames)
