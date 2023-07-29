from torch.optim import lr_scheduler


def fetch_scheduler(optimizer, CFG, is_finetuning, n_train_batches):
    # @TODO: test other schedulers than cosine
    epochs = CFG.finetuning_epochs if is_finetuning else CFG.pretraining_epochs
    if CFG.scheduler == "cosine":
        T_max = int(n_train_batches * epochs)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=CFG.min_lr
        )  # , last_epoch=epochs-1)
    elif CFG.scheduler == "cosine_warm_restarts":
        virt_train_batches = min(n_train_batches, 10000 // CFG.train_batch_size)
        # start by restarting after one epoch, double the restarting rate
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=virt_train_batches,  # limit the virtual dataset size to 10k samples
            eta_min=CFG.min_lr,
            T_mult=2,
            verbose=False,
        )
    elif CFG.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=6,
            threshold=5e-4,
            min_lr=CFG.min_lr,
        )
    elif CFG.scheduler == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif CFG.scheduler == None:
        scheduler = None

    if  CFG.scheduler is not None and CFG.scheduler_warmup_iters > 0:
        wmup_scheduler = lr_scheduler.LinearLR(optimizer,start_factor= 0.01, end_factor=1.0, total_iters= CFG.scheduler_warmup_iters, verbose=False)
        scheduler = lr_scheduler.SequentialLR(optimizer, [wmup_scheduler, scheduler] ,milestones=[CFG.scheduler_warmup_iters], verbose=False)    

    return scheduler
