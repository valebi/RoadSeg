
import datetime
import logging
import time
from argparse import Namespace

from roadseg.train_single_ds import evaluate_finetuning, pretrain_model
from roadseg.utils.args import parse_args
from roadseg.utils.augmentations import get_albumentations
from roadseg.utils.utils import finalize, setup
from roadseg.model.smp_models import build_model
from roadseg.datasets.dataloaders import get_dataloaders
from roadseg.utils.plots import plot_batch



def main(CFG: Namespace):
    """Main function."""
    start = time.time()

    transforms = get_albumentations(CFG)
    train_loader, val_loader, test_splits, n_train_samples = get_dataloaders(CFG, transforms)

    imgs, msks = next(iter(train_loader))
    plot_batch(imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="train", log_dir=CFG.log_dir)
    imgs, msks = next(iter(test_splits[0][1]))
    plot_batch(imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="comp", log_dir=CFG.log_dir)

    model = build_model(CFG, num_classes=2)

    model = pretrain_model(CFG, model, train_loader, val_loader, n_train_samples=n_train_samples)

    avg_f1 = evaluate_finetuning(model, test_splits, CFG, n_comp_samples=144)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Elapsed time: {elapsed}")

    finalize(CFG)


if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        finalize(args)
        raise e
