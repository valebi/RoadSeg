import glob
import logging
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
    ##Added resize to match the training size
    imgs = [
        np.array(
            Image.open(os.path.join(CFG.test_imgs_dir, f)).resize(
                (CFG.img_size, CFG.img_size), Image.Resampling.BILINEAR
            )
        )[:, :, :3]
        for f in img_files
    ]

    if CFG.partial_diffusion:
        # add (un)known mask part
        imgs = [np.concatenate((i, np.zeros_like(i[:, :, :2])), axis=-1) for i in imgs]

    model.to(CFG.device)
    model.eval()

    dirname = os.path.join(CFG.out_dir, f"fold-{fold}")
    os.makedirs(dirname, exist_ok=True)
    imgs = np.asarray(imgs).transpose([0, 3, 1, 2]).astype(np.float32)
    imgs /= 255.0

    imgs = torch.tensor(imgs, device=CFG.device, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(imgs), batch_size=CFG.val_batch_size, shuffle=False
    )

    pred = torch.concat([model(d) for d, in dl], axis=0)
    pred = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()
    pred = pred[:, road_class, :, :] * 255
    pred = pred.astype(np.uint8)
    for i, prd in enumerate(pred):
        img = PIL.Image.fromarray(prd).resize(
            (400, 400), Image.Resampling.NEAREST
        )  # Added resize to match the actual size
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
    masks_to_submission(CFG.submission_file, "", *image_filenames)
    try:
        import kaggle

        kaggle.api.competition_submit(
            file_name=CFG.submission_file,
            message=f"autosubmit: {CFG.experiment_name}",
            competition="ethz-cil-road-segmentation-2023",
        )
        logging.info("Submitted output to kaggle")
    except Exception as e:
        logging.info("Failed to submit to kaggle")
        logging.info(str(e))
