import datetime
import gc
import logging
import pathlib
import time
from argparse import Namespace
import glob


from torch import nn
from torchsummary import summary

from roadseg.datasets.dataloaders import get_dataloaders
from roadseg.model.smp_models import build_model

# @TODO fix imports
# from roadseg.train_single_ds_diffusion import evaluate_finetuning, pretrain_model
# from roadseg.inference_diffusion import make_ensemble, make_submission
from roadseg.utils.augmentations import get_albumentations
from roadseg.utils.mask_to_submission import masks_to_submission
from roadseg.utils.plots import plot_batch
from roadseg.utils.utils import download_file_from_google_drive, finalize, setup
from tta import apply_tta


def main(CFG: Namespace):
    """Main function."""
    if CFG.use_diffusion:
        from roadseg.inference_diffusion import make_ensemble, make_submission
        from roadseg.train_single_ds_diffusion import evaluate_finetuning, pretrain_model
    else:
        from roadseg.inference import make_ensemble, make_submission
        from roadseg.train_single_ds import evaluate_finetuning, pretrain_model

    start = time.time()

    transforms = get_albumentations(CFG)
    train_loader, val_loader, test_splits = get_dataloaders(CFG, transforms)

    # TODO: We may add the link as an arugment too but https://drive.google.com/file/d/1HvnM02Zimq_DspftGFz5ziPD-HWWhERL/view?usp=drive_link cannot be parsed correctly as input
    if CFG.model_download_drive_id:
        print(CFG.model_download_drive_id)
        file_id = CFG.model_download_drive_id  # .split("/")[-2]
        print(file_id)
        destination = CFG.initial_model
        pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model from {CFG.model_download_drive_id} to {destination}")

        download_file_from_google_drive(file_id, destination)
        logging.info(f"Downloaded model to {destination}.")


    if CFG.discriminator_download_drive_id:
        file_id = CFG.discriminator_download_drive_id
        destination = "discriminator.pth"
        pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading discriminator from {CFG.discriminator_download_drive_id} to {destination}")

        download_file_from_google_drive(file_id, destination)
        logging.info(f"Downloaded discriminator to {destination}.")

    if not(CFG.no_pretrain and CFG.no_finetune):
        model = build_model(CFG, num_classes=2).to(CFG.device)
        
    if CFG.debug:
        imgs, msks = next(iter(train_loader))
        plot_batch(
            imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="train", log_dir=CFG.log_dir
        )
        imgs, msks = next(iter(test_splits[0][1]))
        plot_batch(
            imgs[:16].detach().numpy(), msks[:16].detach().numpy(), src="comp", log_dir=CFG.log_dir
        )
        del imgs, msks
        summary(model, input_size=imgs.shape[1:], device=CFG.device)

    gc.collect()  ##Might be useful to garbage collect before we start training
    if not CFG.no_pretrain:
        logging.info(f"Training on {len(train_loader)*CFG.train_batch_size} samples")
        model = pretrain_model(CFG, model, train_loader, val_loader)
        gc.collect()  ##Might be useful to garbage collect before we start fine tuning

    if not CFG.no_finetune:
        logging.info(f"Finetuning on {len(test_splits[0][0])*CFG.train_batch_size} samples")
        avg_score = evaluate_finetuning(model, test_splits, CFG)
        logging.info(f"Average {CFG.metrics_to_watch[0]} scores of folds: {avg_score}.")

    if CFG.tta:
        apply_tta(CFG)

    make_ensemble(CFG)
    image_filenames = sorted(glob.glob(f"{CFG.out_dir}/ensemble/*.png"))
    masks_to_submission(CFG.submission_file, "", *image_filenames)

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
        print(f"Exception {e} occurred, saving a checkpoint...")
        finalize(args)
        raise e
