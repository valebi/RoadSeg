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
    def __init__(self, transforms, img_size, max_samples=-1, add_partial_labels=False):
        super().__init__()
        self.transforms = transforms
        self.is_train = False
        self.crop = None
        self.img_paths = []
        self.lbl_paths = []
        self.max_samples = max_samples
        self.add_partial_labels = add_partial_labels
        if add_partial_labels:
            self.label_mask_crop = A.Compose(
                [
                    A.augmentations.geometric.rotate.Rotate(limit=180, p=0.5, crop_border=False),
                    A.augmentations.crops.transforms.RandomResizedCrop(
                        img_size,
                        img_size,
                        scale=(0.33, 0.33),
                        ratio=(0.85, 1.15),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                ],
                p=1,
            )

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

        if self.add_partial_labels:
            # create random mask from 2x2 grid and provide partial labels accordingly
            s = img.shape[0]
            if np.random.rand() < 0.8:
                grid = np.random.rand(2, 2)
                label_visible_mask = np.zeros((2 * s, 2 * s, 1))  # labels and mask
                for i in range(2):
                    for j in range(2):
                        if grid[i, j] < 0.4:  # 40% chance of being visible
                            label_visible_mask[i * s : (i + 1) * s, j * s : (j + 1) * s] = 1
                label_visible_mask = self.label_mask_crop(image=label_visible_mask)["image"]
                partial_label = label_visible_mask * lbl[:, :, :1]
            else:
                label_visible_mask = np.zeros((s, s, 1))
                partial_label = np.zeros((s, s, 1))
            img = np.concatenate((img, partial_label * 255, label_visible_mask * 255), axis=2)

            """
            print(img.shape, lbl.shape)
            import matplotlib.pyplot as plt

            plt.title("In dataloader")
            plt.subplot(1, 4, 1)
            plt.imshow(img[:,:,:3].astype(int))
            plt.subplot(1, 4, 2)
            plt.imshow(img[:,:,3])
            plt.subplot(1, 4, 3)
            plt.imshow(img[:,:,4])
            plt.subplot(1, 4, 4)
            plt.imshow(lbl[:,:,0])
            plt.show()
            """

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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
        self.img_paths = glob(
            CFG.data_dir + "/ethz-cil-road-segmentation-2023/training/images/*.png"
        )
        self.lbl_paths = [f.replace("images", "groundtruth") for f in self.img_paths]
        if transforms is not None:
            self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
                CFG.img_size,
                CFG.img_size,
                scale=(0.85, 1.15),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
            )
        elif CFG.img_size != 400:
            self.crop = A.augmentations.geometric.resize.Resize(
                CFG.img_size,
                CFG.img_size,
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            )
        self._ensure_size()


class BingDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
        lbls = glob(CFG.data_dir + "/processed-bing-dataset/processed_label/*.png")
        self.img_paths = [
            f
            for f in glob(CFG.data_dir + "/processed-bing-dataset/processed_sat/*.png")
            if f.replace("processed_sat", "processed_label") in lbls
        ]
        self.lbl_paths = [f.replace("processed_sat", "processed_label") for f in self.img_paths]
        """
        self.crop = A.augmentations.crops.transforms.RandomResizedCrop(
            CFG.img_size,
            CFG.img_size,
            scale=(0.85, 1.15),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
        )
        """
        self.crop = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=CFG.img_size,
                    min_width=CFG.img_size,
                    p=1.0,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.augmentations.geometric.rotate.Rotate(
                    limit=180, p=0.25, border_mode=cv2.BORDER_CONSTANT
                ),
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


class RoadtracingDataset(SegmentationDataset):
    def __init__(self, CFG, transforms=None, max_samples=-1):
        # @TODO load the useless ones too
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
        self.img_paths = glob(CFG.data_dir + "/google-roadseg/sat/*/*.png")
        self.lbl_paths = [f.replace("sat", "road") for f in self.img_paths]
        self.crop = A.Compose(
            [
                A.augmentations.geometric.rotate.Rotate(limit=180, p=0.75, crop_border=True),
                A.augmentations.crops.transforms.RandomResizedCrop(
                    CFG.img_size,
                    CFG.img_size,
                    scale=(0.5, 0.9),
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
        super().__init__(
            transforms, CFG.img_size, max_samples, add_partial_labels=CFG.partial_diffusion
        )
        self.size = CFG.img_size
        self.max_margin = max_margin if max_margin != -1 else CFG.img_size // 2
        patch_only = self.transforms is None and not CFG.partial_diffusion
        if patch_only:
            self.max_margin = 0
        self.train_paths = sorted(
            glob(CFG.data_dir + "/ethz-cil-road-segmentation-2023/training/images/*.png")
        )
        self.test_paths = sorted(
            glob(CFG.data_dir + "/ethz-cil-road-segmentation-2023/test/images/*.png")
        )
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
            "roadseg/utils/img1.csv",
            delimiter=",",
            dtype=np.int32,
        )
        self.lookup2 = np.loadtxt(
            "roadseg/utils/img2.csv",
            delimiter=",",
            dtype=np.int32,
        )

        self.add_t0_mask = CFG.partial_diffusion

        self.img1, loc_dict1 = self.assemble_image(self.lookup1, CFG.img_size)
        self.img2, loc_dict2 = self.assemble_image(self.lookup2, CFG.img_size)

        self.loc_dict = {
            **{k: (1, loc_dict1[k]) for k in loc_dict1},
            **{k: (2, loc_dict2[k]) for k in loc_dict2},
        }

        if not patch_only:
            self.crop = A.Compose(
                [
                    A.augmentations.geometric.transforms.PadIfNeeded(
                        CFG.img_size + 2 * self.max_margin,
                        CFG.img_size + 2 * self.max_margin,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    A.augmentations.geometric.rotate.Rotate(limit=180, p=0.5, crop_border=False),
                    A.augmentations.crops.transforms.RandomResizedCrop(
                        CFG.img_size,
                        CFG.img_size,
                        scale=(0.25, 0.4),
                        ratio=(0.85, 1.15),
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
        margin_x_before = min(i, self.max_margin)
        margin_x_after = min(self.img1.shape[0] - i - 1, self.max_margin)
        margin_y_before = min(j, self.max_margin)
        margin_y_after = min(self.img1.shape[1] - j - 1, self.max_margin)
        # cut out (potentially bigger) patch around original location
        _img = self.img1 if img_nr == 1 else self.img2
        patch = _img[
            i - margin_x_before : i + margin_x_after + self.size,
            j - margin_y_before : j + margin_y_after + self.size,
        ]

        if self.add_t0_mask:
            # make sure the center (target tile) is noise (and has actual timestep)
            # and the borders are ground truth (and have timestep 0)
            t0_mask = np.ones_like(patch[:, :, 4:]) * 255  # has shape (patch_h, patch_w, 1)
            t0_mask[
                margin_x_before : margin_x_before + self.size,
                margin_y_before : margin_y_before + self.size,
            ] = 0
            # if we don't know the label we want noise not maximal certainty
            t0_mask[patch[:, :, 4:5] == 0] = 0
            patch = np.concatenate([patch, t0_mask], axis=-1)

        # split masks and images
        # we have (R,G,B, label. loss_mask, t0_mask) = (R, G, B, LABEL, IS_LABEL_KNOWN, IS PART OF TARGET TILE)
        img, lbl = patch[:, :, :3], patch[:, :, 3:]

        # crop to correct size
        if self.crop is not None:
            if np.random.rand() < 0.66:
                aug = self.crop(image=img, mask=lbl)
                img, lbl = aug["image"], aug["mask"]
                # attenuate aliasing artefacts on mask
            else:
                img = img[
                    margin_x_before : margin_x_before + self.size,
                    margin_y_before : margin_y_before + self.size,
                ]
                lbl = lbl[
                    margin_x_before : margin_x_before + self.size,
                    margin_y_before : margin_y_before + self.size,
                ]

            lbl[:, :, 1] = (lbl[:, :, 1] > 124) * 255
        # get into channels_first format
        img, lbl = torch.transpose(torch.tensor(img), 0, 2), torch.transpose(
            torch.tensor(lbl), 0, 2
        )

        img, lbl = (
            img / 255,
            (lbl / 255).type(torch.uint8),
        )  # scale to 0-1 and remove channel dim from mask

        # @TODO make this not break on diffusion
        if self.add_t0_mask:
            if np.random.rand() < 0.5:
                partial_input_mask = lbl[1:2] * lbl[2:]  # known and not part of target tile
            elif np.random.rand() < 0.5:
                partial_input_mask = lbl[1:2] * (1 - lbl[2:])  # known and part of target tile
            else:
                partial_input_mask = torch.zeros_like(lbl[:1])  # no hints at all
            img, lbl = (
                torch.cat([img, lbl[:1] * partial_input_mask, partial_input_mask], dim=0),
                lbl[:2],
            )

        """
        print(img.shape, lbl.shape)
        import matplotlib.pyplot as plt

        plt.title("In dataloader")
        plt.subplot(1, 4, 1)
        plt.imshow(img[:3].permute(1, 2, 0))
        plt.subplot(1, 4, 2)
        plt.imshow(lbl.permute(1, 2, 0)[:, :, 0])
        plt.subplot(1, 4, 3)
        plt.imshow(lbl.permute(1, 2, 0)[:, :, 1])
        if len(lbl) > 2:
            plt.subplot(1, 4, 4)
            plt.imshow(lbl.permute(1, 2, 0)[:, :, 2])
        plt.show()
        """

        return img, lbl


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
