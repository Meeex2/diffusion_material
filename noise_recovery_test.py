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
        # Start from middle of diffusion
        t = torch.tensor([500], device=device)

        # Recover noise
        with torch.no_grad():
            noise = model.recover_noise(samples[i : i + 1], t, num_steps=num_steps)
            recovered_noise.append(noise)

            # Calculate statistics for this sample
            stats = {
                "mean": noise.mean().item(),
                "std": noise.std().item(),
                "min": noise.min().item(),
                "max": noise.max().item(),
            }
            noise_stats.append(stats)

            # Compare with original noise
            orig_noise = original_noise[i : i + 1]
            comparison = {
                "mean_diff": abs(noise.mean() - orig_noise.mean()).item(),
                "std_diff": abs(noise.std() - orig_noise.std()).item(),
                "mse": torch.mean((noise - orig_noise) ** 2).item(),
                "cosine_sim": torch.nn.functional.cosine_similarity(
                    noise.view(1, -1), orig_noise.view(1, -1)
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
