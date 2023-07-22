from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from torchsummary import summary

from roadseg.model.lucidrains_medsegdiff import *

img_size = 256

# summary(model, input_size=(3, img_size, img_size), device="cpu")


class OurDiffuser(nn.Module):
    def __init__(
        self,
        smp_encoder_name="efficientnet-b5",
        smp_encoder_weights=None,
        smp_in_channels=3,
        smp_encoder_depth=5,
    ):
        super().__init__()
        self.encoder = get_encoder(
            smp_encoder_name,
            in_channels=smp_in_channels,
            depth=smp_encoder_depth,
            weights=None,
        )

        self.encoder.requires_grad_(False)  # freeze encoder weights

    def load_encoder_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        try:
            filtered_keys = []
            for k, v in state_dict.items():
                if not k.startswith("module.encoder."):
                    continue
                filtered_keys.append((k.replace("module.encoder.", ""), v))

            state_dict = OrderedDict(filtered_keys)
            self.encoder.load_state_dict(state_dict, strict=True)

        except:
            raise AttributeError(
                f"Model weights loading failed. Please initialize the model with the same paramaters used in initial training."
            )

        self.adapter_blocks = nn.ModuleList()
        # self.adapter_blocks.append(nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False))
        # assert len(diffuser.encoder._blocks) == len(self.adapter_blocks)

    def forward(self, x):
        # stages = self.encoder.get_stages() ## @TODO:Many of the encoders work with this, but not all. I need to determine the supported encoder types.

        features = self.encoder(x)
        print(f"Start of encoder: {x.shape}")
        for i in range(len(features)):
            print(f"End of stage {i}: {features[i].shape}")

        # x = self.encoder._conv_stem(x)
        # x = self.encoder._bn0(x)
        # x = self.encoder._swish(x)

        # for enc_blk in self.encoder._blocks:
        #     print(f"Start of block {i}: {x.shape}")
        #     x = enc_blk(x)

        # for stage in stages:
        #     print(f"Start of block {i}: {x.shape}")
        #     x = stage(x)
        #     i+=1

        print("End of encoder\n")


@beartype
class OurUnet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        mask_channels=1,
        input_img_channels=3,
        init_dim=None,
        out_dim=None,
        dim_mults: tuple = (1, 2, 2, 4),
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head=16,
        attn_heads=2,
        mid_transformer_depth=1,
        self_condition=False,
        resnet_block_groups=4,
        conditioning_klass=Conditioning,
        skip_connect_condition_fmaps=False,  # whether to concatenate the conditioning fmaps in the latter decoder upsampling portion of unet
        encoding_model=None,
    ):
        super().__init__()

        self.debug = False
        self.image_size = image_size

        # determine dimensions

        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition

        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding=3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention related params

        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads)

        # layers

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        self.conditioners = nn.ModuleList([])

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps

        if encoding_model is not None:
            self.pretrained_encoder = encoding_model
            self.pretrained_encoder.requires_grad_(False)  # freeze encoder weights
            self.encoding_model_adapter_dim_ins = (
                self.pretrained_encoder.out_channels
            )  # First one is the input image

        self.encoding_model_adapter_blocks = nn.ModuleList([])

        # @TODO : We may add a sanity check for the dimensions of the pretrained encoder and the unet

        # downsampling encoding blocks

        self.downs = nn.ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(conditioning_klass(curr_fmap_size, dim_in))

            if ind < len(self.encoding_model_adapter_dim_ins):
                self.encoding_model_adapter_blocks.append(
                    nn.Conv2d(
                        dim_in + self.encoding_model_adapter_dim_ins[ind],
                        dim_in,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(attn_klass(dim_in, **attn_kwargs)),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

            if not is_last:
                curr_fmap_size //= 2

        # middle blocks

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = Transformer(mid_dim, depth=mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # condition encoding path will be the same as the main encoding path

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # upsampling decoding blocks

        self.ups = nn.ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(
            zip(reversed(in_out), reversed(full_self_attn))
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmaps else 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                        Residual(attn_klass(dim_out, **attn_kwargs)),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # projection out to predictions

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(self, x, time, cond, x_self_cond=None):
        dtype, skip_connect_c = x.dtype, self.skip_connect_condition_fmaps

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        features = self.pretrained_encoder(cond) if self.pretrained_encoder else None
        x = self.init_conv(x)
        r = x.clone()

        c = self.cond_init_conv(cond)

        t = self.time_mlp(time)

        h = []
        i = 0
        for (
            (block1, block2, attn, downsample),
            (cond_block1, cond_block2, cond_attn, cond_downsample),
            conditioner,
        ) in zip(self.downs, self.cond_downs, self.conditioners):
            if features and i < len(self.encoding_model_adapter_dim_ins):
                c = self.encoding_model_adapter_blocks[i](torch.cat([features[i], c], dim=1))
            if self.debug:
                print(f"Start of loop idx {i}, x : {x.shape} , c: {c.shape}")
            x = block1(x, t)
            c = cond_block1(c, t)
            if self.debug:
                print(f"After Block 1 of loop idx {i}, x : {x.shape} , c: {c.shape}")
            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)
            if self.debug:
                print(f"After Block 2 of loop idx {i}, x : {x.shape} , c: {c.shape}")
            x = attn(x)
            c = cond_attn(c)

            # condition using modulation of fourier frequencies with attentive map
            # you can test your own conditioners by passing in a different conditioner_klass , if you believe you can best the paper

            c = conditioner(x, c)

            h.append([x, c] if skip_connect_c else [x])
            if self.debug:
                print(f"Before downsampling of loop idx {i}, x : {x.shape} , c: {c.shape}")
            x = downsample(x)
            c = cond_downsample(c)
            if self.debug:
                print(f"After downsampling(End) of loop idx {i}, x : {x.shape} , c: {c.shape}")
                print(f"-------------------\n")
            i += 1

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        x = (
            x + c
        )  # seems like they summed the encoded condition to the encoded input representation

        x = self.mid_transformer(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == "__main__":
    diffuser = OurDiffuser()
    encoding_model = diffuser.encoder
    # diffuser.load_encoder_weights("temp_folder/chk.bin")
    unet = OurUnet(
        dim=64,
        image_size=img_size,
        dim_mults=(1, 2, 4, 8, 16),
        full_self_attn=(False, False, False, False, False),
        mask_channels=1,
        input_img_channels=3,
        self_condition=False,
        encoding_model=encoding_model,
    )
    b = 2
    import torchsummary

    lbl = torch.randn((b, 1, img_size, img_size))
    img = torch.randn((b, 3, img_size, img_size))
    time = torch.randint(0, 100, (b,), dtype=torch.long)
    torchsummary.summary(unet, input_size=[lbl.shape[1:], time.shape[1:], img.shape[1:]])

    model_out = unet(lbl, time, img, None)
