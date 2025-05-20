"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from scipy.stats import betaprime
# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulations
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


def sample(net, num_samples, dim, num_steps=50, device="cuda:0"):
    """Generate samples using deterministic DDIM sampling."""
    # Initialize latents with noise
    latents = torch.randn([num_samples, dim], device=device)

    # Initialize scheduler parameters (exactly as in inverse_stable_diffusion.py)
    timesteps = torch.arange(num_steps, device=device)
    alpha_t = torch.cos((timesteps / num_steps) * torch.pi / 2)
    sigma_t = torch.sin((timesteps / num_steps) * torch.pi / 2)
    lambda_t = torch.log(alpha_t / sigma_t)

    # DDIM sampling process
    for t in reversed(range(num_steps)):
        s = t
        prev_t = t - 1 if t > 0 else 0

        # Get scheduler parameters
        lambda_s, lambda_t = lambda_t[s], lambda_t[prev_t]
        sigma_s, sigma_t = sigma_t[s], sigma_t[prev_t]
        alpha_s, alpha_t = alpha_t[s], alpha_t[prev_t]
        h = lambda_t - lambda_s
        phi_1 = torch.expm1(-h)

        # Get noise prediction
        with torch.no_grad():
            noise_pred = net(latents, torch.tensor([s], device=device))
            model_s = (
                noise_pred  # In our case, model output is already noise prediction
            )

            # DDIM update (exactly as in inverse_stable_diffusion.py)
            latents = (sigma_s / sigma_t) * (latents + alpha_t * phi_1 * model_s)

    return latents


def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(data) if augment_pipe is not None else (data, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(
        self,
        denosie_fn,
        data,
        labels=None,
        augment_pipe=None,
        stf=False,
        pfgmpp=False,
        ref_data=None,
    ):
        if pfgmpp:
            # N,
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(
                a=self.N / 2.0, b=self.D / 2.0, size=data.shape[0]
            ).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1 - 1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma**2
            y, augment_labels = (
                augment_pipe(data) if augment_pipe is not None else (data, None)
            )
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma**2
            y, augment_labels = (
                augment_pipe(data) if augment_pipe is not None else (data, None)
            )
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


class EDMLoss:
    def __init__(
        self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100, gamma=5, opts=None
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss


def inverse_sample(net, x, num_steps=50, device="cuda:0"):
    """Inverse diffusion process to recover latents from generated samples."""
    # Initialize scheduler parameters (exactly as in inverse_stable_diffusion.py)
    timesteps = torch.arange(num_steps, device=x.device)
    alpha_t = torch.cos((timesteps / num_steps) * torch.pi / 2)
    sigma_t = torch.sin((timesteps / num_steps) * torch.pi / 2)
    lambda_t = torch.log(alpha_t / sigma_t)

    # Initialize latents with the input
    latents = x.clone()

    # Inverse DDIM process
    for t in range(num_steps):
        s = t
        next_t = t + 1 if t < num_steps - 1 else num_steps - 1

        # Get scheduler parameters
        lambda_s, lambda_t = lambda_t[s], lambda_t[next_t]
        sigma_s, sigma_t = sigma_t[s], sigma_t[next_t]
        alpha_s, alpha_t = alpha_t[s], alpha_t[next_t]
        h = lambda_t - lambda_s
        phi_1 = torch.expm1(-h)

        # Get noise prediction
        with torch.no_grad():
            noise_pred = net(latents, torch.tensor([s], device=device))
            model_s = (
                noise_pred  # In our case, model output is already noise prediction
            )

            # Inverse DDIM update (exactly as in inverse_stable_diffusion.py)
            latents = (sigma_t / sigma_s) * (latents - alpha_t * phi_1 * model_s)

    return latents


def fixedpoint_correction(
    x, s, t, x_t, net, order=1, n_iter=500, step_size=0.1, th=1e-3
):
    """Fixed-point correction algorithm for improving inverse diffusion results."""
    input = x.clone()
    original_step_size = step_size

    # Initialize scheduler parameters (exactly as in inverse_stable_diffusion.py)
    num_steps = 20  # Should match the number of steps used in generation
    timesteps = torch.arange(num_steps, device=x.device)
    alpha_t = torch.cos((timesteps / num_steps) * torch.pi / 2)
    sigma_t = torch.sin((timesteps / num_steps) * torch.pi / 2)
    lambda_t = torch.log(alpha_t / sigma_t)

    # Get scheduler parameters
    s_idx = s.item() if isinstance(s, torch.Tensor) else s
    t_idx = t.item() if isinstance(t, torch.Tensor) else t
    lambda_s, lambda_t = lambda_t[s_idx], lambda_t[t_idx]
    sigma_s, sigma_t = sigma_t[s_idx], sigma_t[t_idx]
    alpha_s, alpha_t = alpha_t[s_idx], alpha_t[t_idx]
    h = lambda_t - lambda_s
    phi_1 = torch.expm1(-h)

    # Pre-compute constants
    sigma_ratio = sigma_t / sigma_s
    alpha_phi = alpha_t * phi_1

    for i in range(n_iter):
        # Get noise prediction
        with torch.no_grad():
            noise_pred = net(input, t)
            # Calculate predicted x_t (exactly as in inverse_stable_diffusion.py)
            x_t_pred = sigma_ratio * input - alpha_phi * noise_pred

            # Calculate loss
            loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction="sum")

            if loss.item() < th:
                break

            # Forward step method
            input = input - step_size * (x_t_pred - x_t)

            # Clear unnecessary tensors
            del noise_pred, x_t_pred
            torch.cuda.empty_cache()

    return input
