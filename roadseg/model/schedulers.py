from torch.optim import lr_scheduler


def fetch_scheduler(optimizer, CFG, epochs, n_train_batches):
    # @TODO: test other schedulers than cosine
    # @TODO: make this depend on whether pretraining or finetuning
    if CFG.scheduler == "cosine":
        # @TODO: find better heuristic for T_max
        T_max = int(n_train_batches * epochs)
        T_max = max(T_max, 100)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=CFG.min_lr
        )  # , last_epoch=epochs-1)
    elif CFG.scheduler == "cosine_warm_restarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_train_batches, eta_min=CFG.min_lr, last_epoch=epochs - 1, T_mult=1.5
        )
    elif CFG.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
            threshold=0.0001,
            metric="val_loss",
            min_lr=CFG.min_lr,
        )
    elif CFG.scheduler == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None

    return scheduler