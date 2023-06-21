from torch.optim import lr_scheduler

def fetch_scheduler(optimizer, CFG, n_train_samples):
    if CFG.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=25, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'cosine_warm_restarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=2,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduler == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler