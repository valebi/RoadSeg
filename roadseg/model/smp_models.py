from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_model(CFG, num_classes):

    if CFG.smp_model != "Unet":
        raise NotImplementedError(f"Model {CFG.smp_model} not implemented.")

    # @TODO: the UNet does not contract fully with depth=4. But if it does we need img_size % 32 == 0
    init_weights =  "imagenet" if CFG.smp_encoder_init_weights else None
 
    model = smp.Unet(
        encoder_name=CFG.smp_backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
        decoder_channels=(64, 64, 32, 16),
        encoder_depth=4,
    )
    
    if CFG.initial_model: 
        try:
            state_dict = torch.load(CFG.initial_model)
            filtered_keys =[]
            for k,v in state_dict.items():
                filtered_keys.append( (k.replace("module.", "") , v))
            state_dict = OrderedDict(filtered_keys)
            model.load_state_dict(state_dict, strict=True)
        except:
            raise AttributeError(f"Model weights loading failed. Please initialize the model with the same paramaters used in initial training.")


    model.to(CFG.device)
    model = nn.DataParallel(model)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
