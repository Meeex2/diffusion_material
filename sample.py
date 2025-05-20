import torch
import argparse
import warnings
import time
import numpy as np
from tqdm import tqdm

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings("ignore")


def generate_samples(
    model, num_samples, sample_dim, mean, device, batch_size=1024, num_steps=20
):
    """Generate samples using deterministic DDIM sampling in batches."""
    all_samples = []
    all_noise = []  # Keep track of the noise used

    # Start with a smaller batch size and adjust based on memory
    current_batch_size = min(batch_size, 256)  # Start with smaller batches

    for i in tqdm(range(0, num_samples, current_batch_size), desc="Generating samples"):
        try:
            batch_size_curr = min(current_batch_size, num_samples - i)

            # Generate samples using DDIM sampling
            with torch.no_grad():
                x_next = sample(
                    model.denoise_fn_D, batch_size_curr, sample_dim, num_steps=num_steps
                )
                x_next = x_next * 2 + mean.to(device)

            # Move to CPU and clear GPU memory
            all_samples.append(x_next.cpu())
            del x_next
            torch.cuda.empty_cache()

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

    return torch.cat(all_samples, dim=0).to(device)


def reconstruct_latents(model, samples, num_steps=50, n_iter=500, batch_size=256):
    """Reconstruct latent representations from generated samples in batches."""
    all_latents = []
    num_samples = samples.shape[0]

    # Calculate optimal batch size based on available memory
    # Start with a smaller batch size and increase if memory allows
    current_batch_size = min(batch_size, 128)  # Start with smaller batches

    for i in tqdm(
        range(0, num_samples, current_batch_size), desc="Reconstructing latents"
    ):
        try:
            batch_size_curr = min(current_batch_size, num_samples - i)
            batch_samples = samples[i : i + batch_size_curr]

            # Process batch with memory optimization
            latents = model.reconstruct_latent(
                batch_samples, num_steps=num_steps, n_iter=n_iter
            )

            # Move to CPU and clear GPU memory
            all_latents.append(latents.cpu())
            del latents
            torch.cuda.empty_cache()

            # Try to increase batch size if memory allows
            if i + current_batch_size >= num_samples:
                current_batch_size = min(current_batch_size * 2, batch_size)

        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM, reduce batch size and retry
                current_batch_size = max(current_batch_size // 2, 32)
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return torch.cat(all_latents, dim=0)


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    inverse_path = args.inverse_path if hasattr(args, "inverse_path") else None
    batch_size = args.batch_size if hasattr(args, "batch_size") else 1024

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{ckpt_path}/model.pt"))

    # Generate samples
    start_time = time.time()
    num_samples = train_z.shape[0]
    sample_dim = in_dim

    # Generate samples in batches using DDIM sampling
    x_next = generate_samples(
        model,
        num_samples,
        sample_dim,
        mean,
        device,
        batch_size=batch_size,
        num_steps=steps,
    )

    # If inverse path is provided, perform inverse diffusion
    if inverse_path:
        print("Performing inverse diffusion...")
        reconstructed_latents = reconstruct_latents(
            model,
            x_next,
            num_steps=steps,
            n_iter=500,
            batch_size=batch_size // 4,  # Use smaller batch size for inverse diffusion
        )
        np.save(inverse_path, reconstructed_latents.cpu().numpy())
        print(f"Saved reconstructed latents to {inverse_path}")

    # Convert samples to tabular data
    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse, args.device
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print("Time:", end_time - start_time)
    print("Saving sampled data to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="sample.csv")
    parser.add_argument(
        "--inverse_path",
        type=str,
        default=None,
        help="Path to save reconstructed latents",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for processing samples",
    )
    args = parser.parse_args()
    main(args)
