from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tabsyn.diffusion_utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

        # Initialize scheduler parameters
        self._init_scheduler_params()

    def _init_scheduler_params(self):
        """Initialize scheduler parameters for inverse diffusion."""
        num_steps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers to ensure parameters are moved to the correct device
        self.register_buffer("alpha_t", alphas_cumprod)
        self.register_buffer(
            "sigma_t", torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        )
        self.register_buffer("lambda_t", torch.log(self.sigma_t))

    def scale_model_input(self, x, t):
        """Scale the model input according to the noise level."""
        return x

    def convert_model_output(self, noise_pred, t, x):
        """Convert model output to denoised sample."""
        return noise_pred

    def forward(self, x, sigma):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        gamma=5,
        opts=None,
        pfgmpp=False,
    ):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(
            P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None
        )

    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle missing scheduler parameters."""
        # Create a new state dict without the scheduler parameters
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(x in k for x in ["alpha_t", "sigma_t", "lambda_t"])
        }

        # Load the filtered state dict
        super().load_state_dict(filtered_state_dict, strict=False)

        # Initialize scheduler parameters if they don't exist
        if not hasattr(self.denoise_fn_D, "alpha_t"):
            self.denoise_fn_D._init_scheduler_params()

    def round_sigma(self, sigma):
        """Round sigma values to valid range."""
        return self.denoise_fn_D.round_sigma(sigma)

    def inverse_diffusion(self, x, num_steps=50):
        """
        Perform inverse diffusion to recover latents from generated samples.
        """
        from tabsyn.diffusion_utils import inverse_sample

        return inverse_sample(self.denoise_fn_D, x, num_steps)

    def reconstruct_latent(self, x, num_steps=50, n_iter=500):
        """
        Reconstruct latent representation from generated sample using inverse diffusion
        and fixed-point correction.
        """
        from tabsyn.diffusion_utils import fixedpoint_correction

        # First perform inverse diffusion
        latents = self.inverse_diffusion(x, num_steps)

        # Apply fixed-point correction
        s = torch.tensor(0, device=x.device)
        t = torch.tensor(num_steps - 1, device=x.device)
        x_t = x

        corrected_latents = fixedpoint_correction(
            latents, s, t, x_t, self.denoise_fn_D, order=1, n_iter=n_iter, step_size=0.1
        )

        return corrected_latents

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()
