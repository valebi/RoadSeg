"""Utility functions and classes. @TODO might have license issues"""

import argparse
import datetime
import logging
import os
import random
from glob import glob

import numpy as np
import requests
import torch
import wandb

from roadseg.utils.args import parse_args


def set_seed(seed: int):
    """Set seed for deterministic behavior. Runs nondeterministically if -1.

    @TODO: might have license issues
    Args:
        seed (int): Seed to use.
    """
    if seed == -1:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup() -> argparse.Namespace:
    """Setup arguments and logging.

    Returns:
        argparse.Namespace: Parsed arguments from command line.
    """

    cfg = parse_args()

    if cfg.debug:
        # Hardcode some arguments for faster debugging
        cfg.experiment_tag = "debug"
        cfg.pretraining_epochs = 2
        cfg.finetuning_epochs = 2
        cfg.max_per_dataset = 64

    set_seed(cfg.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.experiment_name = f"{cfg.smp_model}_{cfg.smp_backbone}_{cfg.experiment_tag}"
    cfg.log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, timestamp)
    # cfg.out_dir = os.path.join(cfg.out_dir, cfg.experiment_name, timestamp)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(os.path.join(cfg.log_dir, "weights"))

    setup_logging(cfg)

    logging.debug("Command line arguments:")
    for arg, val in vars(cfg).items():
        logging.debug(f"\t{arg}: {val}")
    return cfg


def setup_logging(cfg: argparse.Namespace):
    """Setup logging.

    Args:
        cfg (argparse.Namespace): Parsed arguments.
    """
    log_file = os.path.join(cfg.log_dir, "log.txt") if cfg.log_to_file else None
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG if cfg.debug else logging.INFO,
        filename=log_file,
        force=True if not cfg.log_to_file else False,
    )

    # Suppress logging from other modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    prepare_wandb(cfg)


def finalize(CFG: argparse.Namespace):
    """Cleanup logging.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    if CFG.wandb:
        wandb.finish()

    if len(os.listdir(CFG.log_dir)) == 0:
        os.rmdir(CFG.log_dir)


def prepare_wandb(CFG):
    wandb_mode = "disabled" if (not CFG.wandb) else "online"
    # @TODO: add id as cmd line argument, make this able to resume.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="topoflop",
        project=CFG.wandb_project_name,
        resume="allow",
        name=CFG.experiment_name + "__" + timestamp,
        mode=wandb_mode,
        config=vars(CFG),
        dir=CFG.log_dir,
    )


def log_info(CFG: argparse.Namespace, info: dict, src="", step=None, epoch=None):  ##NOT USED
    """
    Log the given dictionary info, appends src to all fields.
    """
    if not CFG.wandb:
        return
    for key, value in info.items():
        if step:
            wandb.log({src + key: value}, step=step)
        else:
            wandb.log({src + key: value})


def log_images(imgs, msks, preds):
    """
    Accepts
        imgs: BxCxHxW numpy array
        msks: BxHxW numpy array
        preds: BxHxW numpy array
    """
    class_labels = {
        0: "x",
        1: "road",
    }
    MAX_NUM_OF_IMAGES = 5
    logs = []
    for im, mask, pred in zip(imgs, msks, preds):
        mask = mask.round().astype(np.uint8)
        pred = pred.round().astype(np.uint8)
        i = wandb.Image(
            im.transpose([1, 2, 0]),
            masks={
                "predictions": {"mask_data": pred, "class_labels": class_labels},
                "ground_truth": {"mask_data": mask, "class_labels": class_labels},
            },
        )
        logs.append(i)
        if len(logs) >= MAX_NUM_OF_IMAGES:
            break

    wandb.log({"samples": logs})


def download_file_from_google_drive(file_id, destination, chunk_size=16384):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    params = {"id": file_id, "confirm": 1}
    response = session.get(url, params=params, stream=True)

    with open(destination, "wb") as f:
        for i, chunk in enumerate(response.iter_content(chunk_size)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # yield i, chunk_size_ #This can be used to implement a progress bar
