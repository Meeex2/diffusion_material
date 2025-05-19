import torch
import argparse
import warnings
import time
import numpy as np

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings("ignore")


def generate_samples(model, num_samples, sample_dim, mean, device):
    """Generate samples using the diffusion model."""
    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)
    return x_next


def reconstruct_latents(model, samples, num_steps=50, n_iter=500):
    """Reconstruct latent representations from generated samples."""
    return model.reconstruct_latent(samples, num_steps=num_steps, n_iter=n_iter)


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    inverse_path = args.inverse_path if hasattr(args, "inverse_path") else None

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

    # Generate samples
    x_next = generate_samples(model, num_samples, sample_dim, mean, device)

    # If inverse path is provided, perform inverse diffusion
    if inverse_path:
        print("Performing inverse diffusion...")
        reconstructed_latents = reconstruct_latents(model, x_next)
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
    args = parser.parse_args()
    main(args)
