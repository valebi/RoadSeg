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
def pred_from_dataloader(model, dl, device, num_ensemble=2, road_class=1):
    model.to(device)
    model.eval()
    preds = []
    for imgs, masks in dl:
        # ensembled prediction
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        pred = np.mean(
            [
                model.sample(imgs.to(device)).cpu().detach().numpy()[:, road_class]
                for i in range(num_ensemble)
            ],
            axis=0,
        )
        preds.append(pred)
    return torch.tensor(np.concatenate(preds, axis=0))


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
    imgs = np.asarray(imgs).transpose([0, 3, 1, 2]).astype(np.float32)
    imgs /= 255.0

    imgs = torch.tensor(imgs, device=CFG.device, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(imgs), batch_size=CFG.val_batch_size, shuffle=False
    )

    model.to(CFG.device)
    model.eval()
    preds = []
    for (d,) in dl:
        # ensembled prediction

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        pred = np.mean(
            [model.sample(d).cpu().detach().numpy()[:, road_class] for i in range(5)], axis=0
        )
        preds.append(pred)
    pred = np.concatenate(preds, axis=0)

    pred = pred * 255
    pred = pred.astype(np.uint8)

    dirname = os.path.join(CFG.out_dir, f"fold-{fold}")
    os.makedirs(dirname, exist_ok=True)
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
