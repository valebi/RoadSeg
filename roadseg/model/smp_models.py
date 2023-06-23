import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_model(CFG, num_classes):
    if CFG.smp_model != "Unet":
        raise NotImplementedError(f"Model {CFG.smp_model} not implemented.")

    # @TODO: the UNet does not contract fully with depth=4. But if it does we need img_size % 32 == 0
    init_weights =  "imagenet" if CFG.smp_init_weights else None
 
    model = smp.Unet(
        encoder_name=CFG.smp_backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
        decoder_channels=(64, 64, 32, 16),
        encoder_depth=4,
    )
    model.to(CFG.device)
    model = nn.DataParallel(model)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
