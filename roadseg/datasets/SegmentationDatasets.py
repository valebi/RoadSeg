import logging
from argparse import Namespace
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import imageio as io
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, max_samples=-1):
        super().__init__()
        self.transforms = transforms
        self.is_train = False
        self.label_transform = None
        self.crop = None
        self.img_paths = []
        self.msk_paths = []
        self.max_samples = max_samples

    def __len__(self):
        return len(self.img_paths)

    def _ensure_size(self):
        self.img_paths = np.array(self.img_paths)
        self.msk_paths = np.array(self.msk_paths)
        if self.max_samples != -1 and self.max_samples < len(self.img_paths):
            ixes = np.random.choice(
                np.arange(0, len(self.img_paths)), self.max_samples, replace=False
            )
            self.img_paths = self.img_paths[ixes]
            self.msk_paths = self.msk_paths[ixes]

    def __getitem__(self, index):
        ##TODO:CAN WE CHANGE THIS TO IMAGEIO, opencv reads images in BGR order, might have issues with it later since it is unconventional
        # img = cv2.imread(self.img_paths[index])
        # msk = cv2.imread(self.msk_paths[index])
        img = io.imread(self.img_paths[index])[:, :, :3]
        msk = io.imread(self.msk_paths[index])

        # logging.info(img.shape, msk.shape, img.dtype, msk.dtype, img.max(), msk.max(), img.min(), msk.min())

        # remove / reorder labels to map them to 0 = BG, 255 = ROAD
        if self.label_transform is not None:
            msk = self.label_transform(msk)

        # treat masks as single-channel images (for augmentations)
        if len(np.asarray(msk).shape) == 2:
            msk = np.expand_dims(msk, -1)
        else:
            msk = np.asarray(msk)

        # bring to correct size
        if self.crop is not None:
            aug = self.crop(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        # augment
        if self.is_train and self.transforms is not None:
            aug = self.transforms(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        # get into channels_first format
        img, msk = torch.transpose(torch.tensor(img), 0, 2), torch.transpose(
            torch.tensor(msk), 0, 2
        )
        return (
            img / 255,
            (msk / 255).type(torch.uint8)[0],
        )  # scale to 0-1 and remove channel dim from mask


class MaptilerDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "/maptiler-custom-tiles/maptiler_tiles_processed/images/*/*/*.jpg"
        )
        self.msk_paths = [
            f.replace("images", "masks").replace(".jpg", ".png") for f in self.img_paths
        ]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()

    def label_transform(self, msk):
        return (msk == 0).astype(np.uint8) * 255


class HofmannDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(CFG.data_dir + "/roadseg-download-openstreetmap/images/*.png")
        self.msk_paths = [f.replace("images", "labels") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.3, 0.5),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()

    def label_transform(self, msk):
        return 255 - msk[:, :, 2:]


class ESRIDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "//esri-streetmap-tiles/esri_tiles_processed/images/*/*/*.jpg"
        )
        self.msk_paths = [
            f.replace("images", "masks").replace(".jpg", ".png") for f in self.img_paths
        ]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()

    def label_transform(self, msk):
        return (msk == 0).astype(np.uint8) * 255


class CIL23Dataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "/ethz-cil-road-segmentation-2023/training/images/*.png"
        )
        self.msk_paths = [f.replace("images", "groundtruth") for f in self.img_paths]
        if transforms is not None or CFG.img_size != 400:
            self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
                CFG.img_size,
                CFG.img_size,
                scale=(0.85, 1.15),
                ratio=(0.75, 1.3333333333333333),
                interpolation=cv2.INTER_LINEAR,
            )
        self._ensure_size()


class BingDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        lbls = glob(CFG.data_dir + "/bingscrape-noarrow/bing/label/*.png")
        self.img_paths = [
            f
            for f in glob(CFG.data_dir + "/bingscrape-noarrow/bing/sat/*.png")
            if f.replace("sat", "label") in lbls
        ]
        self.msk_paths = [f.replace("sat", "label") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()


dataset_map = {
    "hofmann": HofmannDataset,
    "cil": CIL23Dataset,
    "maptiler": MaptilerDataset,
    "esri": ESRIDataset,
    "bing": BingDataset,
}
