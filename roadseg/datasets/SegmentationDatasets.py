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
        self.crop = None
        self.img_paths = []
        self.lbl_paths = []
        self.max_samples = max_samples

    def __len__(self):
        return len(self.img_paths)

    def _ensure_size(self):
        self.img_paths = np.array(self.img_paths)
        self.lbl_paths = np.array(self.lbl_paths)
        if self.max_samples != -1 and self.max_samples < len(self.img_paths):
            ixes = np.random.choice(
                np.arange(0, len(self.img_paths)), self.max_samples, replace=False
            )
            self.img_paths = self.img_paths[ixes]
            self.lbl_paths = self.lbl_paths[ixes]

    def __getitem__(self, index):
        ##TODO:CAN WE CHANGE THIS TO IMAGEIO, opencv reads images in BGR order, might have issues with it later since it is unconventional
        # img = cv2.imread(self.img_paths[index])
        # lbl = cv2.imread(self.lbl_paths[index])
        img = io.imread(self.img_paths[index])[:, :, :3]
        lbl = io.imread(self.lbl_paths[index])

        # logging.info(img.shape, lbl.shape, img.dtype, lbl.dtype, img.max(), lbl.max(), img.min(), lbl.min())

        # remove / reorder labels to map them to 0 = BG, 255 = ROAD
        loss_mask = self.loss_mask_transform(lbl)
        _lbl = self.label_transform(lbl)
        lbl = np.stack([_lbl, loss_mask], axis=2)

        # bring to correct size
        if self.crop is not None:
            aug = self.crop(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]

        # augment
        if self.is_train and self.transforms is not None:
            aug = self.transforms(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]

        # get into channels_first format
        img, lbl = torch.transpose(torch.tensor(img), 0, 2), torch.transpose(
            torch.tensor(lbl), 0, 2
        )

        """
        print(img.shape, lbl.shape)
        import matplotlib.pyplot as plt

        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(lbl.permute(1, 2, 0)[:, :, 0])
        plt.show()
        """

        return (
            img / 255,
            (lbl / 255).type(torch.uint8),
        )  # scale to 0-1 and remove channel dim from mask

    def label_transform(self, lbl):
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        return lbl

    def loss_mask_transform(self, lbl):
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        return np.ones_like(lbl) * 255


class MaptilerDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "/maptiler-custom-tiles/maptiler_tiles_processed/images/*/*/*.jpg"
        )
        self.lbl_paths = [
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

    def label_transform(self, lbl):
        return (lbl == 0).astype(np.uint8) * 255


class HofmannDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(CFG.data_dir + "/roadseg-download-openstreetmap/images/*.png")
        self.lbl_paths = [f.replace("images", "labels") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.4, 0.6),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()

    def label_transform(self, lbl):
        return 255 - lbl[:, :, 0]


class ESRIDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "//esri-streetmap-tiles/esri_tiles_processed/images/*/*/*.jpg"
        )
        self.lbl_paths = [
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

    def label_transform(self, lbl):
        return (lbl == 0).astype(np.uint8) * 255

    def loss_mask_transform(self, lbl):
        return (1 - (lbl == 204).astype(np.uint8)) * 255


class CIL23Dataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(transforms, max_samples)
        self.img_paths = glob(
            CFG.data_dir + "/ethz-cil-road-segmentation-2023/training/images/*.png"
        )
        self.lbl_paths = [f.replace("images", "groundtruth") for f in self.img_paths]
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
        self.lbl_paths = [f.replace("sat", "label") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()


class CleanBingDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        # @TODO load the useless ones too
        super().__init__(transforms, max_samples)
        lbls = glob(CFG.data_dir + "/processed-bing-dataset/processed_label/*.png")
        self.img_paths = [
            f
            for f in glob(CFG.data_dir + "/processed-bing-dataset/processed_sat/*.png")
            if f.replace("processed_sat", "processed_label") in lbls
        ]
        self.lbl_paths = [f.replace("processed_sat", "processed_label") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()


class RoadtracingDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        # @TODO load the useless ones too
        super().__init__(transforms, max_samples)
        self.img_paths = glob(CFG.data_dir + "/roadtracing/processed/images/*.png")
        self.lbl_paths = [f.replace("images", "groundtruth") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()


class EPFLDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        # @TODO load the useless ones too
        super().__init__(transforms, max_samples)
        self.img_paths = glob(CFG.data_dir + "/epfl-roadseg/processed/images/*.png")
        self.lbl_paths = [f.replace("images", "groundtruth") for f in self.img_paths]
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
        )
        self._ensure_size()

    def label_transform(self, lbl):
        return (lbl > 150).astype(np.uint8) * 255


class GoogleDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        # @TODO load the useless ones too
        super().__init__(transforms, max_samples)
        self.img_paths = glob(CFG.data_dir + "/google-roadseg/sat/*/*.png")
        self.lbl_paths = [f.replace("sat", "road") for f in self.img_paths]
        self.crop = A.Compose(
            [
                A.augmentations.geometric.rotate.Rotate(limit=180, p=0.75, crop_border=True),
                A.augmentations.geometric.transforms.Flip(p=0.5),
                A.augmentations.crops.transforms.RandomResizedCrop(
                    CFG.img_size,
                    CFG.img_size,
                    scale=(0.85, 1.15),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_LINEAR,
                ),
            ],
            p=1,
        )
        self._ensure_size()

    def label_transform(self, lbl):
        return lbl.astype(np.uint8)


dataset_map = {
    "hofmann": HofmannDataset,
    "cil": CIL23Dataset,
    "maptiler": MaptilerDataset,
    "esri": ESRIDataset,
    "bing": BingDataset,
    "bing-clean": CleanBingDataset,
    "roadtracing": RoadtracingDataset,
    "epfl": EPFLDataset,
    "google": GoogleDataset,
}
