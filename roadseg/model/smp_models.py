import logging
from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
from torch import nn

import roadseg.model.dummy_unet as dummy_unet


def build_model(CFG, num_classes):

    if CFG.smp_backbone == "dummy-unet":
        return dummy_unet.build_model(CFG, num_classes)

    if CFG.smp_model != "Unet":
        raise NotImplementedError(f"Model {CFG.smp_model} not implemented.")

   
    init_weights =  "imagenet" if CFG.smp_encoder_init_weights else None
    
    # @TODO: the UNet does not contract fully with depth=4. But if it does we need img_size % 32 == 0
    model = smp.Unet(
        encoder_name= "efficientnet-b5",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
        decoder_channels=(1024, 512, 256, 64, 16),
        encoder_depth=5,
    )
    
    if CFG.initial_model: 
        try:
            state_dict = torch.load(CFG.initial_model)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights file {CFG.initial_model} not found.")
        
        try:
            filtered_keys =[]
            for k,v in state_dict.items():
                filtered_keys.append( (k.replace("module.", "") , v))
            state_dict = OrderedDict(filtered_keys)
            model.load_state_dict(state_dict, strict=True)
        except:
            raise AttributeError(f"Model weights loading failed. Please initialize the model with the same paramaters used in initial training.")
        logging.info(f"Model weights loaded from {CFG.initial_model}.")
        del state_dict, filtered_keys


    model.to(CFG.device)
    model = nn.DataParallel(model)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
