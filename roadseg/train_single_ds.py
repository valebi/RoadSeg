"""functionality to train a model on a single dataset or a combination of datasets treated as single dataset"""

import copy
import gc
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from roadseg.model.metrics import (
    BCELoss,
    dice_coef,
    f1_loss,
    iou_coef,
    precision,
    recall,
    reg_f1_loss,
)
from roadseg.model.optimizers import fetch_scheduler
from roadseg.utils.plots import plot_batch
from roadseg.utils.utils import log_images


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, criterion):
    n_accumulate = 1  # max(1, 32//CFG.train_batch_size) @TODO: what is this?
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % n_accumulate == 0:
            # xm.optimizer_step(optimizer, barrier=True)
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
        )
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, optimizer, device, epoch, criterion):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Valid epoch {epoch}")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # print("Before", y_pred.shape)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)[:, 1]
        val_prec = precision(y_pred.cpu(), masks.cpu())
        val_rec = recall(y_pred.cpu(), masks.cpu())
        val_scores.append([val_prec, val_rec])

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_memory=f"{mem:0.2f} GB"
        )
    val_scores = np.mean(val_scores, axis=0)
    val_scores = [
        val_scores[0],
        val_scores[1],
        (2 * ((val_scores[0] * val_scores[1]) / (val_scores[0] + val_scores[1]))),
    ]
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores, images, y_pred, masks


def run_training(
    model,
    model_name,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs,
    criterion,
    use_wandb,
    log_dir,
    plot_freq=3,
):
    # To automatically log gradients

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_dice      = -np.inf
    # best_f1        = -np.inf
    best_loss = np.inf
    # best_epoch     = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}", end="")
        train_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
        )

        val_loss, val_scores, last_in, last_pred, last_msk = valid_one_epoch(
            model, valid_loader, optimizer, criterion=criterion, device=device, epoch=epoch
        )

        if plot_freq >= 1 and epoch % plot_freq == 1:
            plot_batch(
                last_in.cpu().numpy(),
                last_msk.cpu().numpy(),
                preds=last_pred.cpu().numpy(),
                src=f"val-epoch-{epoch}",
                log_dir=log_dir,
            )
        if use_wandb:
            log_images(last_in.cpu().numpy(), last_msk.cpu().numpy(), last_pred.cpu().numpy())

        val_precision, val_recall, val_f1 = val_scores

        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["Valid F1"].append(val_f1)
        history["Valid Precision"].append(val_precision)
        history["Valid Recall"].append(val_recall)

        if use_wandb:
            global_step = epoch * len(train_loader)
            wandb.log({f"{model_name} Train Loss": train_loss}, step=global_step)
            wandb.log({f"{model_name} Valid Loss": val_loss}, step=global_step)
            wandb.log({f"{model_name} Valid F1": val_f1}, step=global_step)
            wandb.log(
                {f"{model_name} Valid Precision": val_precision, "Valid Recall": val_recall},
                step=global_step,
            )
            wandb.log({f"{model_name} Epoch": epoch}, step=global_step)

        print(
            f"Valid Loss: {val_loss:0.4f} | Valid F1: {val_f1:0.4f} | Valid Precision: {val_precision:0.4f} | Valid Recall: {val_recall:0.4f}"
        )

        # deep copy the model
        if val_loss <= best_loss:
            print(f"Valid Score Improved ({best_loss:0.4f} ---> {val_loss:0.4f})")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(log_dir, "weights", f"best_epoch-{model_name}.bin")
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved under {PATH}")

        PATH = os.path.join(log_dir, "weights", f"last_epoch-{model_name}.bin")
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
        )
    )
    print("Best Score: {:.4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def pretrain_model(CFG, model, train_loader, val_loader, n_train_samples=None):
    model_name = f"pretrain"
    optimizer = optim.Adam(model.parameters(), lr=CFG.pretraining_lr, weight_decay=CFG.weight_decay)
    scheduler = fetch_scheduler(optimizer, CFG, n_train_samples)
    model, history_pre = run_training(
        model,
        model_name,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        criterion=BCELoss,
        device=CFG.device,
        use_wandb=CFG.wandb,
        log_dir=CFG.log_dir,
        num_epochs=CFG.pretraining_epochs,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title("Pretraining: ")
    ax.plot(history_pre["Train Loss"])
    ax.plot(history_pre["Valid Loss"])
    ax.legend(["train loss", "val loss"])
    fig.savefig(os.path.join(CFG.log_dir, f"pretraining_loss.png"))
    plt.show()
    return model


def evaluate_finetuning(pretrained_model, comp_splits, CFG, n_comp_samples=None):
    f1_scores = []
    for fold, (train_loader, val_loader) in enumerate(comp_splits):
        model = copy.deepcopy(pretrained_model)
        model_name = f"finetune-fold-{fold}"
        optimizer = optim.Adam(
            model.parameters(), lr=CFG.finetuning_lr, weight_decay=CFG.weight_decay
        )
        scheduler = fetch_scheduler(optimizer, CFG=CFG, n_train_samples=n_comp_samples)
        model, history = run_training(
            model,
            model_name,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            criterion=reg_f1_loss,
            device=CFG.device,
            use_wandb=CFG.wandb,
            log_dir=CFG.log_dir,
            num_epochs=CFG.finetuning_epochs,
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("Finetuning fold {}".format(fold))
        ax.plot(history["Train Loss"])
        ax.plot(history["Valid Loss"])
        ax.legend(["train loss", "val loss"])
        fig.savefig(os.path.join(CFG.log_dir, f"finetuning_loss_fold_{fold}.png"))
        plt.show()
        f1_scores.append(np.max(history["Valid F1"]))

    print("Best F1 scores after FT: {}".format(np.mean(f1_scores)))
    return np.mean(f1_scores)
