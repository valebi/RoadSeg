from typing import Any, List, Tuple, Union

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification

from .losses import DiceDisc, PatchGANDiscriminatorLoss

# @TODO: CLEANUP
##These all throws errors? Fixed versions below
JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
# BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
BCELoss = nn.CrossEntropyLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)

##Metrics
f1_score = torchmetrics.F1Score(task="binary")
precision = torchmetrics.Precision(task="binary")
recall = torchmetrics.Recall(task="binary")


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    """@TODO add source"""
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


##Throws error
# def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
#     """ @TODO add source"""
#     y_true = y_true.to(torch.float32)
#     y_pred = (y_pred>thr).to(torch.float32)
#     inter = (y_true*y_pred).sum(dim=dim)
#     union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
#     iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
#     return iou


# ported from https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
def f1_loss(y_pred, y_true, eps=1e-10, road_class=1):
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)[:, road_class]
    y_true = y_true.float()

    tp = torch.sum(y_true * y_pred, axis=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = torch.sum((1 - y_true) * y_pred, axis=0)
    fn = torch.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1 = torch.nan_to_num(f1)
    return 1 - torch.mean(f1)


def reg_f1_loss(y_pred, y_true):
    # return 0.5*BCELoss(y_pred, y_true) #+ 0.5*DiceLoss(y_pred, y_true)
    return 0.2 * BCELoss(y_pred, y_true) + 0.8 * f1_loss(
        y_pred, y_true
    )  # + 0.5*DiceLoss(y_pred, y_true)

def patch_f1_loss(y_pred, y_true):
    # return 0.5*BCELoss(y_pred, y_true) #+ 0.5*DiceLoss(y_pred, y_true)
    return 0.2 * get_loss("smp_soft_ce")(y_pred, y_true) + 0.8 * f1_loss(
        torch.nn.functional.avg_pool2d(y_pred, kernel_size=16, stride=16),
        torch.nn.functional.avg_pool2d(y_true.to(torch.float32), kernel_size=16, stride=16),
    )  # + 0.5*DiceLoss(y_pred, y_true)


def get_loss(name: str, device = None):
    loss_dict = {
        "bce": BCELoss,
        "reg_f1": reg_f1_loss,
        "patch_f1": patch_f1_loss,
        "smp_dice": smp.losses.DiceLoss(mode="multiclass"),
        "smp_jaccard": smp.losses.JaccardLoss(mode="multiclass"),
        "smp_lovasz": smp.losses.LovaszLoss(
            mode="multiclass", per_image=False
        ),  ##Per image does not change the result
        "smp_tversky": smp.losses.TverskyLoss(mode="multiclass"),
        "smp_soft_ce": smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1),
        "patchgan_disc" : PatchGANDiscriminatorLoss(discriminator_lr = 0.001, device=device, discriminator_init_weights="discriminator.pth"),
        "patchgan_dice" : DiceDisc(discriminator_lr= 0.00005, device=device, discriminator_init_weights="discriminator.pth"),
    }

    loss = loss_dict.get(name, None)
    if loss is None:
        raise ValueError(f"Loss {name} not found")
    return loss


##ACTUAL Metrics
def get_metrics(names: List[str]):
    metrics_dict = {
        "f1": AccumulatedF1(),
        "precision": AccumulatedPrecision(),
        "recall": AccumulatedRecall(),
        "iou": IOU(),
        "jaccard": IOU(),
        "compf1": CompF1(),
    }

    metric_funcs = []
    for name in names:
        metric = metrics_dict.get(name, None)
        if metric is None:
            raise ValueError(f"Metric {name} not found")
        metric_funcs.append(metric)

    return metric_funcs


class AccumulatedRecall:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def __call__(self, y_pred, y_true):

        with torch.no_grad():
            tp, fp, tn, fn, sup = torchmetrics.classification.BinaryStatScores(
                threshold=0.5, multidim_average="global", ignore_index=None, validate_args=True
            )(y_pred.cpu(), y_true.cpu())
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

        if self.tp + self.fn == 0:
            return torch.tensor(0.0)

        return self.tp / (self.tp + self.fn)


class AccumulatedPrecision:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def __call__(self, y_pred, y_true):

        with torch.no_grad():
            tp, fp, tn, fn, sup = torchmetrics.classification.BinaryStatScores(
                threshold=0.5, multidim_average="global", ignore_index=None, validate_args=True
            )(y_pred.cpu(), y_true.cpu())
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

        if self.tp + self.fp == 0:
            return torch.tensor(0.0)

        return self.tp / (self.tp + self.fp)


class AccumulatedF1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def __call__(self, y_pred, y_true):

        with torch.no_grad():
            tp, fp, tn, fn, sup = torchmetrics.classification.BinaryStatScores(
                threshold=0.5, multidim_average="global", ignore_index=None, validate_args=True
            )(y_pred.cpu(), y_true.cpu())
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        if self.tp + self.fn + self.fp == 0:
            return torch.tensor(0.0)
        return 2 * self.tp / (2 * self.tp + self.fn + self.fp)


class CompF1:
    def __init__(self):
        self.pooling = nn.AvgPool2d(kernel_size=(16, 16), stride=16)
        self.threshold = 0.25
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def __call__(self, y_pred, y_true):
        y_true_p = (self.pooling(y_true.to(torch.float)) > self.threshold).to(torch.long)
        y_pred_p = (self.pooling(y_pred.to(torch.float)) > self.threshold).to(torch.long)

        with torch.no_grad():
            tp, fp, tn, fn, sup = torchmetrics.classification.BinaryStatScores(
                threshold=0.5, multidim_average="global", ignore_index=None, validate_args=True
            )(y_pred_p.cpu(), y_true_p.cpu())

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        if self.tp + self.fn + self.fp == 0:
            return torch.tensor(0.0)
        return 2 * self.tp / (2 * self.tp + self.fn + self.fp)


class IOU:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.iou = 0

    def __call__(self, y_pred, y_true):
        current_batch_size = y_pred.shape[0]
        self.total_samples += current_batch_size

        with torch.no_grad():

            self.iou += current_batch_size * torchmetrics.classification.BinaryJaccardIndex()(
                y_pred.cpu(), y_true.cpu()
            )

        if self.total_samples == 0:
            return torch.tensor(0.0)

        return self.iou / self.total_samples
