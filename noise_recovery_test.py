import torch
import numpy as np
import argparse
from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate
from tabsyn.diffusion_utils import sample
import json
import os


def generate_samples_with_noise(
    model, num_samples, in_dim, mean, device, batch_size=1024
):
    """Generate samples while keeping track of the noise used."""
    all_samples = []
    all_noise = []

    # Start with a smaller batch size and adjust based on memory
    current_batch_size = min(batch_size, 256)  # Start with smaller batches

    for i in tqdm(range(0, num_samples, current_batch_size), desc="Generating samples"):
        try:
            batch_size_curr = min(current_batch_size, num_samples - i)

            # Generate noise
            noise = torch.randn([batch_size_curr, in_dim], device=device)
            all_noise.append(noise)

            # Generate samples using the noise
            with torch.no_grad():
                x_next = sample(model.denoise_fn_D, batch_size_curr, in_dim)
                x_next = x_next * 2 + mean.to(device)
                all_samples.append(x_next)

            # Try to increase batch size if memory allows
            if i + current_batch_size >= num_samples:
                current_batch_size = min(current_batch_size * 2, batch_size)

        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM, reduce batch size and retry
                current_batch_size = max(current_batch_size // 2, 64)
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return torch.cat(all_samples, dim=0), torch.cat(all_noise, dim=0)


def fixedpoint_correction(
    x, s, t, x_t, net, order=1, n_iter=500, step_size=0.1, th=1e-3
):
    """Fixed-point correction algorithm for improving inverse diffusion results."""
    input = x.clone()
    original_step_size = step_size

    # Convert timesteps to indices for accessing scheduler parameters
    s_idx = s.item() if isinstance(s, torch.Tensor) else s
    t_idx = t.item() if isinstance(t, torch.Tensor) else t

    # Pre-compute scheduler parameters
    sigma_s = net.sigma_t[s_idx]
    sigma_t = net.sigma_t[t_idx]
    alpha_s = net.alpha_t[s_idx]
    alpha_t = net.alpha_t[t_idx]
    lambda_s = net.lambda_t[s_idx]
    lambda_t = net.lambda_t[t_idx]
    phi_1 = torch.expm1(-(lambda_t - lambda_s))

    # Pre-compute constants to avoid repeated calculations
    sigma_ratio = sigma_t / sigma_s
    alpha_phi = alpha_t * phi_1

    for i in range(n_iter):
        # Scale input according to noise level
        latent_model_input = input

        # Get noise prediction with memory optimization
        with torch.no_grad():
            noise_pred = net(latent_model_input, t)
            # Calculate predicted x_t
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


def evaluate_noise_recovery(model, samples, original_noise, device, num_steps=20):
    """Evaluate how well the model can recover the original noise."""
    print("\nEvaluating noise recovery...")
    results = {}

    # Initialize lists to store recovered noise
    recovered_noise = []
    noise_stats = []
    comparison_stats = []

    # Process each sample
    for i in tqdm(range(len(samples)), desc="Recovering noise"):
        # Get the sample
        x = samples[i : i + 1]

        # Recover noise using inverse diffusion with fixed-point correction
        with torch.no_grad():
            # First perform inverse diffusion
            s = torch.tensor(0, device=device)
            t = torch.tensor(num_steps - 1, device=device)

            # Get initial estimate using inverse diffusion
            latents = model.inverse_diffusion(x, num_steps=num_steps)

            # Apply fixed-point correction
            recovered_latent = fixedpoint_correction(
                latents, s, t, x, model.denoise_fn_D, order=1, n_iter=500, step_size=0.1
            )

            recovered_noise.append(recovered_latent)

            # Calculate statistics for this sample
            stats = {
                "mean": recovered_latent.mean().item(),
                "std": recovered_latent.std().item(),
                "min": recovered_latent.min().item(),
                "max": recovered_latent.max().item(),
            }
            noise_stats.append(stats)

            # Compare with original noise
            orig_noise = original_noise[i : i + 1]
            comparison = {
                "mean_diff": abs(recovered_latent.mean() - orig_noise.mean()).item(),
                "std_diff": abs(recovered_latent.std() - orig_noise.std()).item(),
                "mse": torch.mean((recovered_latent - orig_noise) ** 2).item(),
                "cosine_sim": torch.nn.functional.cosine_similarity(
                    recovered_latent.view(1, -1), orig_noise.view(1, -1)
                ).item(),
            }
            comparison_stats.append(comparison)

    # Calculate aggregate statistics
    all_recovered = torch.cat(recovered_noise, dim=0)
    results["recovered_stats"] = {
        "mean": all_recovered.mean().item(),
        "std": all_recovered.std().item(),
        "min": all_recovered.min().item(),
        "max": all_recovered.max().item(),
    }

    # Calculate aggregate comparison statistics
    results["comparison_stats"] = {
        "mean_mean_diff": np.mean([s["mean_diff"] for s in comparison_stats]),
        "mean_std_diff": np.mean([s["std_diff"] for s in comparison_stats]),
        "mean_mse": np.mean([s["mse"] for s in comparison_stats]),
        "mean_cosine_sim": np.mean([s["cosine_sim"] for s in comparison_stats]),
    }

    # Print results
    print("\nRecovered noise statistics:")
    print(f"Mean: {results['recovered_stats']['mean']:.4f}")
    print(f"Std: {results['recovered_stats']['std']:.4f}")
    print(f"Min: {results['recovered_stats']['min']:.4f}")
    print(f"Max: {results['recovered_stats']['max']:.4f}")

    print("\nComparison with original noise:")
    print(f"Mean difference: {results['comparison_stats']['mean_mean_diff']:.4f}")
    print(f"Std difference: {results['comparison_stats']['mean_std_diff']:.4f}")
    print(f"Mean MSE: {results['comparison_stats']['mean_mse']:.4f}")
    print(
        f"Mean cosine similarity: {results['comparison_stats']['mean_cosine_sim']:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="noise_recovery_test")
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse = (
        get_input_generate(args)
    )
    in_dim = train_z.shape[1]

    denoise_fn = MLPDiffusion(in_dim, 1024).to(args.device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(args.device)
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))
    model.eval()

    # Generate samples with known noise
    print("\nGenerating samples with known noise...")
    samples, original_noise = generate_samples_with_noise(
        model, args.num_samples, in_dim, train_z.mean(0), args.device, args.batch_size
    )

    # Save samples and original noise
    np.save(os.path.join(args.save_dir, "samples.npy"), samples.cpu().numpy())
    np.save(
        os.path.join(args.save_dir, "original_noise.npy"), original_noise.cpu().numpy()
    )
    print(f"Saved samples and original noise to {args.save_dir}")

    # Evaluate noise recovery
    results = evaluate_noise_recovery(
        model, samples, original_noise, args.device, args.num_steps
    )

    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
