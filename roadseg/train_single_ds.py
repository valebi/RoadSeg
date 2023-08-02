"""functionality to train a model on a single dataset or a combination of datasets treated as single dataset"""

import copy
import gc
import logging
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from roadseg.inference import generate_predictions
from roadseg.model.metrics import get_loss, get_metrics
from roadseg.model.schedulers import fetch_scheduler
from roadseg.utils.plots import plot_batch
from roadseg.utils.utils import log_images


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    epoch,
    criterion,
    use_wandb,
    model_name,
    metric_to_monitor,
    file=None,
):
    n_accumulate = 1  # max(1, 32//CFG.train_batch_size) @TODO: what is this?
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader), file=file, desc=f"Train epoch {epoch}"
    )
    for step, (images, labels) in pbar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        labels, loss_mask = labels[:, 0], labels[:, 1]

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            y_pred = y_pred * loss_mask[:, None]
            labels = labels * loss_mask
            loss = criterion(y_pred, labels)
            loss = loss / n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % n_accumulate == 0:
            # xm.optimizer_step(optimizer, barrier=True)
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                else:
                    if step == 0:
                        scheduler.step(metric_to_monitor)  ##Checks at every epoch

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        if use_wandb and "finetune" not in model_name:
            global_step = step + ((epoch - 1) * len(dataloader))  ##Since epoch starts from 1
            if global_step % 10 == 0:
                wandb.log({f"{model_name}/epoch_loss": epoch_loss}, step=global_step)
                wandb.log({f"{model_name}/lr": optimizer.param_groups[0]["lr"]}, step=global_step)

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
        )
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(
    model, dataloader, optimizer, device, epoch, criterion, metrics_to_watch, file=None
):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = [None] * len(metrics_to_watch)

    for metric in metrics_to_watch:
        if hasattr(metric, "reset"):
            metric.reset()

    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader), file=file, desc=f"Valid epoch {epoch}"
    )
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long)
        labels, loss_mask = masks[:, 0], masks[:, 1]

        batch_size = images.size(0)

        y_pred = model(images)
        y_pred = y_pred * loss_mask[:, None]
        labels = labels * loss_mask
        loss = criterion(y_pred, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # logging.info("Before", y_pred.shape)
        # y_pred = torch.sigmoid(y_pred[:, 1])
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)[:, 1]

        # val_prec = precision(y_pred.cpu(), labels.cpu())
        # val_rec = recall(y_pred.cpu(), labels.cpu())

        for i, metric in enumerate(metrics_to_watch):
            val_scores[i] = metric(y_pred, labels).item()

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_memory=f"{mem:0.2f} GB"
        )

    # val_scores = np.mean(val_scores, axis=0)
    # val_scores = [
    #     val_scores[0],
    #     val_scores[1],
    #     (2 * ((val_scores[0] * val_scores[1]) / (val_scores[0] + val_scores[1]))),
    # ]
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores, images[:, :3], y_pred, labels


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
    metrics_to_watch=None,
    progress_log_file=None,
):
    # To automatically log gradients

    if torch.cuda.is_available():
        logging.info("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_dice      = -np.inf
    # best_f1        = -np.inf
    best_loss = np.inf
    # best_epoch     = -1
    history = defaultdict(list)

    best_score = -np.inf
    metric_to_monitor = -np.inf
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        logging.info(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            use_wandb=use_wandb,
            model_name=model_name,
            metric_to_monitor=metric_to_monitor,
            file=progress_log_file,
        )
        metrics_to_watch_fn = get_metrics(metrics_to_watch)
        val_loss, val_scores, last_in, last_pred, last_msk = valid_one_epoch(
            model,
            valid_loader,
            optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            metrics_to_watch=metrics_to_watch_fn,
            file=progress_log_file,
        )
        metric_to_monitor = val_scores[0]

        if plot_freq >= 1 and epoch % plot_freq == 1:
            plot_batch(
                last_in.cpu().numpy(),
                last_msk.cpu().numpy(),
                pred=last_pred.cpu().numpy(),
                src=f"val-epoch-{epoch}",
                log_dir=log_dir,
            )
        if use_wandb:
            log_images(last_in.cpu().numpy(), last_msk.cpu().numpy(), last_pred.cpu().numpy())

        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        for metric, score in zip(metrics_to_watch, val_scores):
            history[f"Valid {metric}"].append(score)

        if use_wandb:
            # @TODO: Reintroduce global step with offset or similar
            # (cannot reset step but we still want finetuning to be logged)
            log_dict = {
                f"{model_name}/Train Loss": train_loss,
                f"{model_name}/Valid Loss": val_loss,
                f"{model_name}/Epoch": epoch,
            }
            for metric, score in zip(metrics_to_watch, val_scores):
                log_dict[f"{model_name}/Valid {metric}"] = score

            if "finetune" in model_name:
                wandb.log(log_dict)
            else:
                global_step = epoch * len(train_loader)
                wandb.log(log_dict, step=global_step)

        log_str = f"Valid Loss: {val_loss:0.4f}"
        for metric, score in zip(metrics_to_watch, val_scores):
            log_str += f" | Valid {metric}: {score:0.4f}"
        logging.info(f"Epoch {epoch}/{num_epochs} | {log_str}")

        # deep copy the model
        # if val_loss <= best_loss:
        #     logging.info(f"Valid Loss Decreased ({best_loss:0.4f} ---> {val_loss:0.4f})")
        #     best_loss = val_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     PATH = os.path.join(log_dir, "weights", f"best_epoch-{model_name}.bin")
        #     torch.save(model.state_dict(), PATH)
        #     logging.info(f"Model Saved under {PATH}")

        if metric_to_monitor >= best_score:
            logging.info(
                f"{model_name} Model Monitoring Metric {metrics_to_watch[0]} Increased ({best_score:0.4f} ---> {metric_to_monitor:0.4f})"
            )
            best_score = metric_to_monitor
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(log_dir, "weights", f"best_epoch-{model_name}.bin")
            torch.save(model.state_dict(), PATH)
            logging.info(f"Model Saved under {PATH}")

        PATH = os.path.join(log_dir, "weights", f"last_epoch-{model_name}.bin")
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    logging.info(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
        )
    )
    logging.info("Best Monitoring Metric Score: {:.4f}".format(best_score))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def pretrain_model(CFG, model, train_loader, val_loader):
    model_name = f"pretrain"
    optimizer = optim.NAdam(
        model.parameters(), lr=CFG.pretraining_lr, weight_decay=CFG.weight_decay
    )
    scheduler = fetch_scheduler(
        optimizer, CFG, is_finetuning=False, n_train_batches=len(train_loader)
    )
    if CFG.wandb:
        wandb.watch(model, criterion=get_loss(CFG.pretraining_loss, device= CFG.device), log_freq=7000)

    progress_log_file = (
        open(os.path.join(CFG.log_dir, f"{model_name}_progress.log"), "a")
        if CFG.log_to_file
        else None
    )

    model, history_pre = run_training(
        model,
        model_name,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        criterion=get_loss(CFG.pretraining_loss, device= CFG.device),
        device=CFG.device,
        use_wandb=CFG.wandb,
        log_dir=CFG.log_dir,
        num_epochs=CFG.pretraining_epochs,
        metrics_to_watch=CFG.metrics_to_watch,
        progress_log_file=progress_log_file,
    )

    if hasattr(progress_log_file, "close") and callable(progress_log_file.close):
        progress_log_file.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title("Pretraining: ")
    ax.plot(history_pre["Train Loss"])
    ax.plot(history_pre["Valid Loss"])
    ax.legend(["train loss", "val loss"])
    fig.savefig(os.path.join(CFG.log_dir, f"pretraining_loss.png"))
    # plt.show()
    return model


def evaluate_finetuning(pretrained_model, comp_splits, CFG):
    scores_to_watch = []
    for fold, (train_loader, val_loader) in enumerate(comp_splits):
        if CFG.only_fold != -1 and fold != CFG.only_fold:
            continue
        model = copy.deepcopy(pretrained_model)
        model_name = f"finetune-fold-{fold}"
        optimizer = optim.NAdam(
            model.parameters(), lr=CFG.finetuning_lr, weight_decay=CFG.weight_decay
        )
        scheduler = fetch_scheduler(
            optimizer, CFG=CFG, is_finetuning=True, n_train_batches=len(train_loader)
        )
        progress_log_file = (
            open(os.path.join(CFG.log_dir, f"{model_name}_progress.log"), "a")
            if CFG.log_to_file
            else None
        )
        model, history = run_training(
            model,
            model_name,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            criterion=get_loss(CFG.finetuning_loss, device= CFG.device),
            device=CFG.device,
            use_wandb=CFG.wandb,
            log_dir=CFG.log_dir,
            num_epochs=CFG.finetuning_epochs,
            metrics_to_watch=CFG.metrics_to_watch,
            progress_log_file=progress_log_file,
        )
        if hasattr(progress_log_file, "close") and callable(progress_log_file.close):
            progress_log_file.close()

        generate_predictions(model, CFG, fold=fold)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("Finetuning fold {}".format(fold))
        ax.plot(history["Train Loss"])
        ax.plot(history["Valid Loss"])
        ax.legend(["train loss", "val loss"])
        fig.savefig(os.path.join(CFG.log_dir, f"finetuning_loss_fold_{fold}.png"))
        # plt.show()
        scores_to_watch.append(
            np.max(history[f"Valid {CFG.metrics_to_watch[0]}"])
        )  ##Needs to be added back later with monitoring

        gc.collect()

    logging.info(f"Best {CFG.metrics_to_watch[0]} scores after FT: {np.mean(scores_to_watch)}")
    if CFG.wandb:
        wandb.log({f"mean-{CFG.metrics_to_watch[0]}": np.mean(scores_to_watch)})

    return np.mean(scores_to_watch)
