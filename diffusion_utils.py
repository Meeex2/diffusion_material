"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from scipy.stats import betaprime
# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
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
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


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
    """
    Inverse diffusion process to recover latents from generated samples.
    Based on the PRC-Watermark implementation but adapted for tabular data.
    """
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = x.clone()

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = inverse_sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


def inverse_sample_step(net, num_steps, i, t_cur, t_next, x_next):
    """
    Single step of inverse diffusion process.
    """
    x_cur = x_next

    # Increase noise temporarily
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

    # Inverse Euler step
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def fixedpoint_correction(
    x, s, t, x_t, net, order=1, n_iter=500, step_size=0.1, th=1e-3
):
    """
    Fixed-point correction algorithm for improving inverse diffusion results.
    Adapted from PRC-Watermark for tabular data.
    """
    input = x.clone()
    original_step_size = step_size

    for i in range(n_iter):
        latent_model_input = input
        latent_model_input = net.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = net(latent_model_input, s)
        model_output = net.scheduler.convert_model_output(noise_pred, s, input)

        x_t_pred = (net.scheduler.sigma_t[t] / net.scheduler.sigma_t[s]) * input - (
            net.scheduler.alpha_t[t]
            * torch.expm1(-(net.scheduler.lambda_t[t] - net.scheduler.lambda_t[s]))
        ) * model_output

        loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction="sum")

        if loss.item() < th:
            break

        # Forward step method
        input = input - step_size * (x_t_pred - x_t)

    return input
