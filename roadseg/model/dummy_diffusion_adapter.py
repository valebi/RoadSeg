import torch
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead, SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

from roadseg.model.lucidrains_medsegdiff import SinusoidalPosEmb, Unet


class DiffusionAdapter(Unet):
    def __init__(self, smp_model, diffusion_encoder, img_size, dim=64):
        super().__init__(dim=64, image_size=img_size)
        self.smp_model = smp_model
        self.diffusion_encoder = diffusion_encoder
        self.input_img_channels = 3
        self.mask_channels = 2
        self.self_condition = False
        self.image_size = img_size
        self.skip_connect_condition_fmaps = None
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self._freeze(self.smp_model)

    def _freeze(self, module):
        for child in module.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x, time, cond, x_self_cond=None, t0_mask=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        smp_model = self.smp_model
        smp_model.check_input_shape(cond)
        encoder_features = smp_model.encoder(cond)

        # generate features from diffusion input and timestep
        time_features = self.time_mlp(time)
        time_features = time_features[:, :, None, None].repeat(
            1, 1, self.image_size, self.image_size
        )

        """
        import matplotlib.pyplot as plt
        plt.imshow(x[0,0].cpu().numpy())
        plt.show()
        """

        if t0_mask is not None:
            t0_features = self.time_mlp(torch.zeros_like(time))
            t0_features = t0_features[:, :, None, None].repeat(
                1, 1, self.image_size, self.image_size
            )
            # where t0_mask is 1, set timestep to zero, otherwise use actual time step
            time_features = time_features * (1 - t0_mask[:, None]) + t0_features * t0_mask[:, None]

        # concatenate them with the diffusion input
        x = torch.cat((x, time_features), dim=1)

        # generate diffusion bias
        diffusion_features = self.diffusion_encoder(x)

        # add it! (leave first one, it is the image itself)
        features = encoder_features[:1] + [
            encoder_features[i] + 0 * diffusion_features[i] for i in range(1, len(encoder_features))
        ]
        decoder_output = smp_model.decoder(*features)

        masks = smp_model.segmentation_head(decoder_output)

        if smp_model.classification_head is not None:
            labels = smp_model.classification_head(features[-1])
            return masks, labels

        return masks
