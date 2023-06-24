import datetime
import gc
import logging
import pathlib
import time
from argparse import Namespace

from torchsummary import summary

from roadseg.datasets.dataloaders import get_dataloaders
from roadseg.inference import make_ensemble, make_submission
from roadseg.model.smp_models import build_model
from roadseg.train_single_ds import evaluate_finetuning, pretrain_model
from roadseg.utils.augmentations import get_albumentations
from roadseg.utils.plots import plot_batch
from roadseg.utils.utils import download_file_from_google_drive, finalize, setup


def main(CFG: Namespace):
    """Main function."""
    start = time.time()

    transforms = get_albumentations(CFG)
    train_loader, val_loader, test_splits = get_dataloaders(CFG, transforms)

    #TODO: We may add the link as an arugment too but https://drive.google.com/file/d/1HvnM02Zimq_DspftGFz5ziPD-HWWhERL/view?usp=drive_link cannot be parsed correctly as input
    if CFG.model_download_drive_id:
        print(CFG.model_download_drive_id)
        file_id = CFG.model_download_drive_id #.split("/")[-2]
        print(file_id)
        destination = CFG.initial_model
        pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model from {CFG.model_download_drive_id} to {destination}")

        download_file_from_google_drive(file_id, destination)
        logging.info(f"Downloaded model to {destination}.")

    model = build_model(CFG, num_classes=2)

    imgs, msks = next(iter(train_loader))
    plot_batch(
        imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="train", log_dir=CFG.log_dir
    )
    imgs, msks = next(iter(test_splits[0][1]))
    plot_batch(
        imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="comp", log_dir=CFG.log_dir
    )

    if CFG.debug:
        summary(model, input_size=imgs.shape[1:], device=CFG.device)

    gc.collect() ##Might be useful to garbage collect before we start training
    if not CFG.no_pretrain:
        logging.info(f"Training on {len(train_loader)*CFG.train_batch_size} samples")
        model = pretrain_model(CFG, model, train_loader, val_loader)

    logging.info(f"Finetuning on {len(test_splits[0][0])*CFG.train_batch_size} samples")
    avg_f1 = evaluate_finetuning(model, test_splits, CFG)

    make_ensemble(CFG)

    if CFG.make_submission:
        make_submission(CFG)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Elapsed time: {elapsed}")

    finalize(CFG)


if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        print("Exception occurred, saving a checkpoint...")
        finalize(args)
        raise e
