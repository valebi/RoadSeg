import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from roadseg.datasets.SegmentationDatasets import CIL23Dataset, OnepieceCILDataset, dataset_map


def split(dataset, train=0.8):
    n_train = int(train * len(dataset))
    return torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(42)
    )


def make_train_val(datasets, train=0.8):
    train, val = zip(*map(split, datasets))
    for d in train:
        d.dataset.is_train = True
    return torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(val)


def get_dataloaders(CFG, transforms):
    # pretraining datasets
    datasets = [
        dataset_map[ds](CFG, transforms=transforms, max_samples=CFG.max_per_dataset)
        for ds in CFG.datasets
    ]

    train_dataset, val_dataset = make_train_val(datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        num_workers=CFG.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False if "DeepLab" not in CFG.smp_model else True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.val_batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if not CFG.use_diffusion else False,
        pin_memory=True,
        drop_last=True,
    )

    # k-fold split of finetuning datasets
    if CFG.onepiece:
        comp_dataset = OnepieceCILDataset(CFG, transforms=transforms)
        if CFG.use_diffusion and CFG.partial_diffusion:
            comp_dataset_notransforms = OnepieceCILDataset(CFG, transforms=transforms)
        else:
            comp_dataset_notransforms = OnepieceCILDataset(CFG, transforms=None)
    else:
        comp_dataset = CIL23Dataset(CFG, transforms=transforms)
        comp_dataset_notransforms = CIL23Dataset(CFG, transforms=None)

    comp_splits = []
    kf = KFold(
        n_splits=CFG.n_finetuning_folds,
        random_state=(42 if CFG.seed == -1 else CFG.seed),
        shuffle=True,
    )  # these folds should not change! Even if the rest runs nondeterministically
    for train_index, val_index in kf.split(np.arange(len(comp_dataset))):
        train_subset = Subset(comp_dataset, train_index)
        val_subset = Subset(comp_dataset_notransforms, val_index)
        comp_train_loader = DataLoader(
            train_subset,
            batch_size=CFG.train_batch_size,
            num_workers=CFG.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False if "DeepLab" not in CFG.smp_model else True,
        )
        # val dataset is shuffled to get random plots
        comp_val_loader = DataLoader(
            val_subset,
            batch_size=CFG.val_batch_size,
            num_workers=CFG.num_workers,
            shuffle=True if not CFG.use_diffusion else False,
            pin_memory=True,
            drop_last=False if "DeepLab" not in CFG.smp_model else True,
        )
        comp_splits.append((comp_train_loader, comp_val_loader))
    return train_loader, val_loader, comp_splits
