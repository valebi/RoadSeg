import copy
import logging
from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

import roadseg.model.dummy_unet as dummy_unet
from roadseg.model.dummy_diffusion_adapter import DiffusionAdapter
from roadseg.model.lucidrains_medsegdiff import MedSegDiff, Unet


def try_load_weights(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights file {path} not found.")

    try:
        filtered_keys = []
        for k, v in state_dict.items():
            filtered_keys.append((k.replace("module.", ""), v))
        filtered_keys = OrderedDict(filtered_keys)
        model.load_state_dict(filtered_keys, strict=True)
    except:
        raise AttributeError(
            f"Model weights loading failed. Please initialize the model with the same paramaters used in initial training."
        )
    logging.info(f"Model weights loaded from {path}.")
    del state_dict, filtered_keys
    return model


def build_model(CFG, num_classes):
    init_weights = "imagenet" if CFG.smp_encoder_init_weights else None

    # @TODO: the UNet does not contract fully with depth=4. But if it does we need img_size % 32 == 0
    # @TODO: Encoder depth should be equal to the decoder depth right?
    if CFG.decoder_depth == 4:
        decoder_channels = (64, 64, 32, 16)
        encoder_depth = 4
    elif CFG.decoder_depth == 5:
        if CFG.slim:
            decoder_channels = (256, 128, 64, 32, 32)
        else:
            decoder_channels = (1024, 512, 64, 32, 32)
        encoder_depth = 5
    else:
        raise ValueError("Decoder Depth can only be 4 or 5 for now.")

    if CFG.smp_backbone == "dummy-unet":
        return dummy_unet.build_model(CFG, num_classes)
    elif CFG.use_diffusion:
        if True:
            # dummy version
            CFG.use_diffusion = False
            init_model = CFG.initial_model
            CFG.initial_model = None
            model = build_model(CFG, num_classes)
            loaded_weights = False

            if init_model:
                # try loading the encoder/decoder only
                _copy = copy.deepcopy(model)
                try:
                    model = try_load_weights(model, init_model, device=CFG.device)
                    logging.info(f"Loaded weights of ENCODER/DECODER ONLY")
                    loaded_weights = True
                except:
                    model = _copy

            CFG.use_diffusion = True
            time_dim = 32
            diffusion_encoder = get_encoder(
                CFG.smp_backbone,
                in_channels=time_dim + 2,
                depth=encoder_depth,
                weights=CFG.smp_encoder_init_weights,
                output_stride=model.encoder.output_stride,
            )
            adapter = DiffusionAdapter(
                model, diffusion_encoder, img_size=CFG.img_size, dim=time_dim
            )

            diffusion = MedSegDiff(adapter, timesteps=100, objective="pred_x0")  # 1000

            if init_model and not loaded_weights:
                # try loading the whole model
                diffusion = try_load_weights(diffusion, init_model, device=CFG.device)
                logging.info(f"Loaded weights of ENTIRE MODEL")

            # diffusion = diffusion.to(CFG.device)
            # diffusion = nn.DataParallel(diffusion)
            return diffusion
        else:
            # real version
            from temp import OurDiffuser, OurUnet

            diffuser = OurDiffuser(smp_encoder_name=CFG.smp_backbone)
            encoding_model = diffuser.encoder
            if CFG.initial_model:
                diffuser.load_encoder_weights(CFG.initial_model)
            unet = OurUnet(
                dim=16,
                image_size=CFG.img_size,
                dim_mults=(1, 2, 2, 4, 8),
                full_self_attn=(False, False, False, False, False),
                mask_channels=1,
                input_img_channels=3,
                self_condition=False,
                encoding_model=encoding_model,
            )
            return MedSegDiff(unet, timesteps=100, objective="pred_x0").to(CFG.device)
    elif CFG.smp_model == "Unet":
        model = smp.Unet(
            encoder_name=CFG.smp_backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
            decoder_channels=decoder_channels,
            encoder_depth=encoder_depth,
        )
    elif CFG.smp_model == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=CFG.smp_backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
            decoder_channels=decoder_channels,
            encoder_depth=encoder_depth,
        )
    elif CFG.smp_model == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=CFG.smp_backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=init_weights,  # "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
        )
    else:
        raise NotImplementedError(f"Model {CFG.smp_model} not implemented.")

    if CFG.initial_model and not CFG.use_diffusion:
        model = try_load_weights(model, CFG.initial_model, device=CFG.device)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
