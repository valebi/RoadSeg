import copy
import math
from collections import namedtuple
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.fft import fft2, ifft2
from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1),
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(Attention(dim, dim_head=dim_head, heads=heads)),
                        Residual(FeedForward(dim)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# conditioning class


class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = fft2(x)
        x = x * self.ff_parser_attn_map
        x = ifft2(x).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)


# model


@beartype
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        mask_channels=1,
        input_img_channels=3,
        init_dim=None,
        out_dim=None,
        dim_mults: tuple = (1, 2, 4, 8),
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head=32,
        attn_heads=4,
        mid_transformer_depth=1,
        self_condition=False,
        resnet_block_groups=8,
        conditioning_klass=Conditioning,
        skip_connect_condition_fmaps=False,  # whether to concatenate the conditioning fmaps in the latter decoder upsampling portion of unet
    ):
        super().__init__()

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

        # downsampling encoding blocks

        self.downs = nn.ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(conditioning_klass(curr_fmap_size, dim_in))

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

        x = self.init_conv(x)
        r = x.clone()

        c = self.cond_init_conv(cond)

        t = self.time_mlp(time)

        h = []

        for (
            (block1, block2, attn, downsample),
            (cond_block1, cond_block2, cond_attn, cond_downsample),
            conditioner,
        ) in zip(self.downs, self.cond_downs, self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)

            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)

            x = attn(x)
            c = cond_attn(c)

            # condition using modulation of fourier frequencies with attentive map
            # you can test your own conditioners by passing in a different conditioner_klass , if you believe you can best the paper

            c = conditioner(x, c)

            h.append([x, c] if skip_connect_c else [x])

            x = downsample(x)
            c = cond_downsample(c)

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


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MedSegDiff(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=1.0,
    ):
        super().__init__()

        self.model = model

        self.input_img_channels = self.model.input_img_channels
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.image_size = self.model.image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20))
        )
        register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, c, x_self_cond)
        maybe_clip = partial(torch.clamp, min=0.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = torch.nn.functional.softmax(x_start, dim=1)
            # x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, c, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(0.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, c=c, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = 0.5 + 0.5 * torch.randn_like(x) if t > 0 else 0.5  # no noise if t == 0

        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        img = 0.5 + 0.5 * img  # shift to 0.5-centered distribution
        img = torch.clamp(img, min=0.0, max=1.0)  # clamp to [0,1]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        img = 0.5 + 0.5 * img  # shift to 0.5-centered distribution
        img = torch.clamp(img, min=0.0, max=1.0)  # clamp to [0,1]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, cond_img, self_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)
            noise = 0.5 + 0.5 * noise  # shift to 0.5-centered distribution
            noise = torch.clamp(noise, min=0.0, max=1.0)  # clamp to [0,1]

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, cond_img):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = 0.5 + 0.5 * noise  # shift to 0.5-centered distribution
        noise = torch.clamp(noise, min=0.0, max=1.0)  # clamp to [0,1]
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, t0_mask=None, loss_mask=None):
        # add parameter t0_mask and loss_mask
        # modify noise as product of x_start and t0_mask
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = 0.5 + 0.5 * noise  # shift to 0.5-centered distribution
        noise = torch.clamp(noise, min=0.0, max=1.0)  # clamp to [0,1]
        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        """
        import matplotlib.pyplot as plt
        plt.subplot(1, 4, 1)
        plt.imshow(x_start[0, 0].cpu().numpy())
        if t0_mask is not None:
            plt.subplot(1, 4, 2)
            plt.imshow(t0_mask[0].cpu().numpy())
        if loss_mask is not None:
            plt.subplot(1, 4, 3)
            plt.imshow(loss_mask[0].cpu().numpy())

        if t0_mask is not None:
            plt.subplot(1, 4, 4)
            plt.imshow((
                x * (1 - t0_mask) + (x_start * t0_mask))[0,0].cpu().numpy())
        plt.show()
        """

        if t0_mask is not None:
            x = x * (1 - t0_mask[:, None]) + (x_start * t0_mask[:, None])

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # predicting x_0
                x_self_cond = self.model_predictions(x, t, cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, cond, x_self_cond, t0_mask=t0_mask)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
            model_out = torch.nn.functional.softmax(model_out, dim=1)
            # return F.binary_cross_entropy_with_logits(model_out, x_start, weight=loss_mask)
        elif self.objective == "pred_v":
            raise NotImplementedError(
                "v parameterization not implemented yet. Not sure what to do with the softmax"
            )
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if loss_mask is not None:
            return F.mse_loss(loss_mask[:, None] * model_out, loss_mask[:, None] * target)
        else:
            return F.mse_loss(model_out, target)

    def forward(self, img, cond_img, *args, t0_mask=None, loss_mask=None, **kwargs):
        if img.ndim == 3:
            img = rearrange(img, "b h w -> b 1 h w")

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, "b h w -> b 1 h w")

        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)

        b, c, h, w, device, img_size, img_channels, mask_channels = (
            *img.shape,
            img.device,
            self.image_size,
            self.input_img_channels,
            self.mask_channels,
        )

        assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        assert (
            cond_img.shape[1] == img_channels
        ), f"your input medical must have {img_channels} channels"
        assert (
            img.shape[1] == mask_channels
        ), f"the segmented image must have {mask_channels} channels"

        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        return self.p_losses(
            img, times, cond_img, *args, t0_mask=t0_mask, loss_mask=loss_mask, **kwargs
        )
