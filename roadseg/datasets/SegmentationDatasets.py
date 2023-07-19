import logging
import os
from argparse import Namespace
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import imageio as io
import numpy as np
import torch
from skimage.transform import resize
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
            ratio=(0.9, 1.1),
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
            ratio=(0.9, 1.1),
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
                ratio=(0.9, 1.1),
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
            ratio=(0.9, 1.1),
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
            ratio=(0.9, 1.1),
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
            ratio=(0.9, 1.1),
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
                A.augmentations.crops.transforms.RandomResizedCrop(
                    CFG.img_size,
                    CFG.img_size,
                    scale=(0.65, 1.2),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_LINEAR,
                ),
            ],
            p=1,
        )
        self._ensure_size()

    def label_transform(self, lbl):
        return lbl.astype(np.uint8)


class OnepieceCILDataset(SegmentationDataset):
    def assemble_image(self, lookup, size):
        shape = (lookup.shape[0] * size, lookup.shape[1] * size, 5)
        full_img = np.zeros(shape, dtype=np.uint8)
        loc_dict = {}
        for i in range(lookup.shape[0]):
            for j in range(lookup.shape[1]):
                loc_dict[lookup[i, j]] = (i * size, j * size)
                full_img[i * size : (i + 1) * size, j * size : (j + 1) * size, :3] = (
                    resize(self.imgs[lookup[i, j]], (size, size)) * 255
                )
                full_img[i * size : (i + 1) * size, j * size : (j + 1) * size, 3:] = (
                    resize(self.labels[lookup[i, j]], (size, size)) * 255
                )
        return full_img, loc_dict

    def __init__(self, CFG, transforms=None, max_samples=-1, max_margin=-1):
        super().__init__(transforms, max_samples)
        self.size = CFG.img_size
        self.max_margin = max_margin if max_margin != -1 else CFG.img_size // 2
        if self.transforms is None:
            self.max_margin = 0
        self.train_paths = glob(
            CFG.data_dir + "/ethz-cil-road-segmentation-2023/training/images/*.png"
        )
        self.test_paths = glob(CFG.data_dir + "/ethz-cil-road-segmentation-2023/test/images/*.png")
        self.img_paths = self.train_paths + self.test_paths
        self.label_paths = [file.replace("images", "groundtruth") for file in self.img_paths]
        self.imgs = [io.imread(file)[:, :, :3] for file in self.img_paths]
        self.labels = [io.imread(f) if os.path.isfile(f) else None for f in self.label_paths]
        # extend with loss masks
        self.labels = [
            np.stack([lbl, np.ones_like(lbl) * 255], axis=-1)
            if lbl is not None
            else np.zeros((CFG.img_size, CFG.img_size, 2))
            for lbl in self.labels
        ]

        self.lookup1 = np.loadtxt(
            os.path.join(CFG.data_dir, "ethz-cil-road-segmentation-2023", "img1.csv"),
            delimiter=",",
            dtype=np.int32,
        )
        self.lookup2 = np.loadtxt(
            os.path.join(CFG.data_dir, "ethz-cil-road-segmentation-2023", "img2.csv"),
            delimiter=",",
            dtype=np.int32,
        )

        self.img1, loc_dict1 = self.assemble_image(self.lookup1, CFG.img_size)
        self.img2, loc_dict2 = self.assemble_image(self.lookup2, CFG.img_size)

        self.loc_dict = {
            **{k: (1, loc_dict1[k]) for k in loc_dict1},
            **{k: (2, loc_dict2[k]) for k in loc_dict2},
        }

        if self.transforms is not None:
            self.crop = A.Compose(
                [
                    A.augmentations.geometric.rotate.Rotate(limit=180, p=0.75, crop_border=True),
                    A.augmentations.crops.transforms.RandomResizedCrop(
                        CFG.img_size,
                        CFG.img_size,
                        scale=(0.7, 1.1),
                        ratio=(0.9, 1.1),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                ],
                p=1,
            )

        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(50,50))
        plt.subplot(2, 3, 1)
        plt.imshow(self.img1[:,:,:3])
        plt.subplot(2, 3, 2)
        plt.imshow(self.img1[:,:,3])
        plt.subplot(2, 3, 3)
        plt.imshow(self.img1[:,:,4])
        plt.subplot(2, 3, 4)
        plt.imshow(self.img2[:,:,:3])
        plt.subplot(2, 3, 5)
        plt.imshow(self.img2[:,:,3])
        plt.subplot(2, 3, 6)
        plt.imshow(self.img2[:,:,4])
        plt.tight_layout()
        plt.show()
        """

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        img_nr, (i, j) = self.loc_dict[index]
        # get available space on any side
        margin_x = min(min(i, self.img1.shape[0] - i - 1), self.max_margin)
        margin_y = min(min(j, self.img1.shape[1] - j - 1), self.max_margin)
        # cut out (potentially bigger) patch around original location
        _img = self.img1 if img_nr == 1 else self.img2
        patch = _img[
            i - margin_x : i + margin_x + self.size, j - margin_y : j + margin_y + self.size
        ]

        # split masks and images
        img, lbl = patch[:, :, :3], patch[:, :, 3:]

        # crop to correct size
        if self.crop is not None:
            aug = self.crop(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]
            # attenuate aliasing artefacts on mask
            lbl[:, :, 1] = (lbl[:, :, 1] > 124) * 255

        # get into channels_first format
        img, lbl = torch.transpose(torch.tensor(img), 0, 2), torch.transpose(
            torch.tensor(lbl), 0, 2
        )

        print(img.shape, lbl.shape)
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.imshow(lbl.permute(1, 2, 0)[:, :, 0])
        plt.subplot(1, 3, 3)
        plt.imshow(lbl.permute(1, 2, 0)[:, :, 1])
        plt.show()

        return (
            img / 255,
            (lbl / 255).type(torch.uint8),
        )  # scale to 0-1 and remove channel dim from mask


dataset_map = {
    "hofmann": HofmannDataset,
    "cil": CIL23Dataset,
    "onepiece-cil": OnepieceCILDataset,
    "maptiler": MaptilerDataset,
    "esri": ESRIDataset,
    "bing": BingDataset,
    "bing-clean": CleanBingDataset,
    "roadtracing": RoadtracingDataset,
    "epfl": EPFLDataset,
    "google": GoogleDataset,
}
