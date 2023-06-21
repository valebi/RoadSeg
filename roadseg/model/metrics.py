import torch
import segmentation_models_pytorch as smp
import torchmetrics
import torch.nn as nn

# @TODO: CLEANUP
JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
#BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
BCELoss     = nn.CrossEntropyLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
#F1_score    = torchmetrics.classification.F1()
precision   = torchmetrics.Precision(task="binary")
recall      = torchmetrics.Recall(task="binary")

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    """ @TODO add source"""
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    """ @TODO add source"""
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

# ported from https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
def f1_loss(y_pred, y_true, eps=1e-10, road_class=1):
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)[:,road_class]
    y_true = y_true.float()
    
    tp = torch.sum(y_true*y_pred, axis=0)
    tn = torch.sum((1-y_true)*(1-y_pred), axis=0)
    fp = torch.sum((1-y_true)*y_pred, axis=0)
    fn = torch.sum(y_true*(1-y_pred), axis=0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2*p*r / (p+r+eps)
    f1 = torch.nan_to_num(f1)
    return 1 - torch.mean(f1)


def reg_f1_loss(y_pred, y_true):
    #return 0.5*BCELoss(y_pred, y_true) #+ 0.5*DiceLoss(y_pred, y_true)
    return 0.2*BCELoss(y_pred, y_true) + 0.8*f1_loss(y_pred, y_true) #+ 0.5*DiceLoss(y_pred, y_true)