import torch
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead, SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

from roadseg.model.lucidrains_medsegdiff import SinusoidalPosEmb, Unet


class DiffusionAdapter(nn.Module):
    def __init__(
        self,
        smp_model,
        diffusion_encoder,
        img_size,
        timesteps,
        dim=64,
        self_condition=False,
        embed_time=True,
        combination_method="linear",
    ):
        super().__init__()
        self.smp_model = smp_model
        self.diffusion_encoder = diffusion_encoder
        self.input_img_channels = 3
        self.mask_channels = 2
        self.self_condition = self_condition
        self.image_size = img_size
        self.skip_connect_condition_fmaps = None
        self.timesteps = timesteps
        self.embed_time = embed_time
        self.combination_method = combination_method
        if self.embed_time:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        if self.combination_method == "conv":
            self.combination_convs = nn.ModuleList(
                [nn.Conv2d(2 * chn, chn, 1) for chn in self.smp_model.encoder.out_channels]
            )
        elif self.combination_method == "key_value":
            self.key_convs = nn.ModuleList(
                [nn.Conv2d(chn, chn, 1) for chn in self.smp_model.encoder.out_channels]
            )
            self.value_convs = nn.ModuleList(
                [nn.Conv2d(chn, chn, 1) for chn in self.smp_model.encoder.out_channels]
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

        if self.embed_time:
            time_features = self.time_mlp(time)
            time_features = time_features[:, :, None, None].repeat(
                1, 1, self.image_size, self.image_size
            )
        else:
            time_features = (
                time[:, None, None, None].repeat(1, 1, self.image_size, self.image_size)
                / self.timesteps
            )

        """
        import matplotlib.pyplot as plt
        plt.imshow(x[0,0].cpu().numpy())
        plt.show()
        """

        if t0_mask is not None:
            # @TODO: use loss mask aswell (set to t_max, add argument)
            if self.embed_time:
                t0_features = self.time_mlp(torch.zeros_like(time))
                t0_features = t0_features[:, :, None, None].repeat(
                    1, 1, self.image_size, self.image_size
                )
            else:
                t0_features = torch.zeros_like(time_features)
            # where t0_mask is 1, set timestep to zero, otherwise use actual time step
            time_features = time_features * (1 - t0_mask[:, None]) + t0_features * t0_mask[:, None]

        # concatenate them with the diffusion input
        x = torch.cat((x, time_features), dim=1)

        """
        import matplotlib.pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(x[0,0].cpu().numpy())
        plt.subplot(1,3,2)
        plt.imshow(x[0,1].cpu().numpy())
        plt.subplot(1,3,3)
        plt.imshow(cond[0].cpu().numpy().transpose(1,2,0))
        plt.show()
        """

        # generate diffusion bias
        diffusion_features = self.diffusion_encoder(x)

        # add it! (leave first one, it is the image itself)
        if self.combination_method == "conv":
            features = encoder_features[:1] + [
                torch.cat([encoder_features[i], diffusion_features[i]], dim=1)
                for i in range(1, len(encoder_features))
            ]
            features = features[:1] + [
                self.combination_convs[i](features[i]) for i in range(1, len(features))
            ]
        elif self.combination_method == "key_value":
            keys = [None] + [
                self.key_convs[i](diffusion_features[i]) for i in range(1, len(diffusion_features))
            ]
            values = [None] + [
                self.value_convs[i](diffusion_features[i])
                for i in range(1, len(diffusion_features))
            ]
            features = encoder_features[:1] + [
                encoder_features[i]
                + values[i]
                * torch.nn.functional.cosine_similarity(keys[i], encoder_features[i], dim=1)[
                    :, None
                ]
                for i in range(1, len(keys))
            ]
        else:  # linear
            features = encoder_features[:1] + [
                diffusion_features[i] + encoder_features[i]
                for i in range(1, len(diffusion_features))
            ]
        decoder_output = smp_model.decoder(*features)

        masks = smp_model.segmentation_head(decoder_output)

        if smp_model.classification_head is not None:
            labels = smp_model.classification_head(features[-1])
            return masks, labels

        return masks


class PseudoDiffusionWrapper(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, img):
        assert img.size(1) == 5, "expected 5 input channels: RGB + label + invalidity_mask"
        with torch.no_grad():
            pred1 = self.model1.forward(img[:, :3]).detach_()
        ext_img = torch.cat((img, pred1), dim=1)
        return self.model2.forward(ext_img)
