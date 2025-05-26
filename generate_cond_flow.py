import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabsyn.latent_utils import get_input_generate
from tabsyn.tabular_normalizing_flow import ConditionalTabularFlow, extract_correlation_context

def plot_correlation_matrices(real_corr, fake_corr, save_path=None):
    """Plot real vs generated correlation matrices"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot real correlation matrix
    im0 = axs[0].imshow(real_corr.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axs[0].set_title('Real Data Correlation')
    axs[0].set_xlabel('Column')
    axs[0].set_ylabel('Column')
    
    # Plot fake correlation matrix
    im1 = axs[1].imshow(fake_corr.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].set_title('Generated Data Correlation')
    axs[1].set_xlabel('Column')
    axs[1].set_ylabel('Column')
    
    # Add colorbar
    fig.colorbar(im0, ax=axs[0], shrink=0.8)
    fig.colorbar(im1, ax=axs[1], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved correlation plot to {save_path}")
    else:
        plt.show()

def compute_batch_correlation(data, n_columns):
    """
    Compute correlation matrix from data
    Args:
        data: Input data tensor [batch_size, dim]
        n_columns: Number of columns in the original tabular data
    Returns:
        Correlation matrix [n_columns, n_columns]
    """
    features_per_column = data.shape[1] // n_columns
    
    # Reshape data to get column-wise representation
    reshaped_data = data.view(-1, n_columns, features_per_column)
    
    # Get column means (averaging over the feature dimension)
    column_means = reshaped_data.mean(dim=2)
    
    # Calculate correlation matrix
    corr = torch.corrcoef(column_means.T)
    
    return corr

def main():
    parser = argparse.ArgumentParser(description='Generation with Conditional Normalizing Flow')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to generate. Defaults to dataset size.')
    parser.add_argument('--n_flows', type=int, default=8, help='Number of flow layers.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size.')
    parser.add_argument('--n_columns', type=int, default=15, help='Number of columns in original data.')
    parser.add_argument('--flow_type', type=str, default='nsf_ar', 
                        choices=['nsf_ar', 'nsf_c', 'maf', 'realnvp'], 
                        help='Type of flow to use.')
    parser.add_argument('--model_path', type=str, default='None', help='Path to model checkpoint')
    parser.add_argument('--use_target_corr', action='store_true', 
                        help='Use target correlation from real data instead of random')
    parser.add_argument('--target_corr_scale', type=float, default=1.0, 
                        help='Scale factor for target correlation (1.0 = exact match)')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    # Load model and data
    train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]
    
    # Set default num_samples to match the real dataset size if not specified
    if args.num_samples is None:
        args.num_samples = len(train_z)
        print(f"Number of samples set to match dataset size: {args.num_samples}")
    
    # Create the model
    model = ConditionalTabularFlow(
        dim=in_dim, 
        n_flows=args.n_flows, 
        hidden_dim=args.hidden_dim,
        n_columns=args.n_columns,
        flow_type=args.flow_type
    ).to(args.device)
    model = model.double()  # Convert to double precision
    
    # Load the model
    model_path = args.model_path
    if model_path == 'None':
        # Create model name with hyperparameters
        model_name = f"flow_model_type{args.flow_type}_nflows{args.n_flows}_hdim{args.hidden_dim}"
        # Try to find a model with these hyperparameters
        model_path = f"{ckpt_dir}/{model_name}.pt"
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Trying default path...")
            model_path = f"{ckpt_dir}/flow_model.pt"
    
    model_name = os.path.basename(model_path).replace('.pt', '')
    print(f"Loading model from: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # New format
        print(f"Loading model from checkpoint (epoch {checkpoint['epoch']})")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use saved normalization parameters if available
        if 'mean' in checkpoint and 'std' in checkpoint:
            mean = checkpoint['mean']
            std = checkpoint['std']
        else:
            mean, std = train_z.mean(0), train_z.std(0)
            
        # Use saved number of columns if available
        if 'n_columns' in checkpoint:
            n_columns = checkpoint['n_columns']
        else:
            n_columns = args.n_columns
            
        # Load global correlation if available
        if 'global_corr' in checkpoint:
            global_corr = checkpoint['global_corr']
        else:
            global_corr = compute_batch_correlation(train_z, n_columns)
            
        if 'lambda_corr' in checkpoint:
            print(f"Model was trained with correlation loss weight: {checkpoint['lambda_corr']}")
        if 'loss' in checkpoint:
            print(f"Best loss: {checkpoint['loss']:.6f}")
    else:
        # Old format (direct state_dict)
        print("Loading model from legacy checkpoint format")
        model.load_state_dict(checkpoint)
        mean, std = train_z.mean(0), train_z.std(0)
        n_columns = args.n_columns
        global_corr = compute_batch_correlation(train_z, n_columns)

    model.eval()
    
    # Compute target correlation matrix from real data
    target_corr = compute_batch_correlation(train_z, n_columns)
    print(f"Target correlation matrix shape: {target_corr.shape}")
    
    # Create context from target correlation
    # Extract upper triangular part (excluding diagonal)
    triu_indices = torch.triu_indices(n_columns, n_columns, offset=1)
    flat_corr = target_corr[triu_indices[0], triu_indices[1]]
    
    # Scale correlation if requested
    if args.target_corr_scale != 1.0:
        print(f"Scaling target correlation by factor {args.target_corr_scale}")
        flat_corr = flat_corr * args.target_corr_scale
    
    # Replicate context for all samples
    context = flat_corr.unsqueeze(0).repeat(args.num_samples, 1).to(args.device).double()
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with target correlation...")
    with torch.no_grad():
        # Sample from base distribution with conditioning
        z = torch.randn(args.num_samples, in_dim, device=args.device).double()
        
        # Decode through the flow with context
        samples, _ = model.decode(z, context)
        
        # Apply inverse transformation to get back to original scale
        samples = samples * std.to(args.device) + mean.to(args.device)
    
    # Create output directory with model name
    output_dir = os.path.join(curr_dir, f'results_{args.flow_type}_nflows{args.n_flows}_hdim{args.hidden_dim}', 
                             args.dataname, 'conditional_flow_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save samples with model name in filename
    samples_filename = f'samples_{model_name}_corr{args.target_corr_scale:.1f}.npy'
    np.save(os.path.join(output_dir, samples_filename), samples.cpu().numpy())
    print(f"Saved samples to {output_dir}/{samples_filename}")
    
    # Evaluate correlation preservation
    print("\nEvaluating correlation preservation...")
    
    # Compute correlation matrix of generated samples
    gen_corr = compute_batch_correlation(samples, n_columns)
    
    # Calculate correlation matrix error
    corr_error = torch.nn.functional.mse_loss(target_corr, gen_corr)
    print(f"Correlation matrix MSE: {corr_error.item():.6f}")
    
    # Plot correlation matrices
    plot_path = os.path.join(output_dir, f'correlation_comparison_{model_name}.png')
    plot_correlation_matrices(target_corr, gen_corr, save_path=plot_path)
    
    # Demonstrate invertibility (optional)
    if args.flow_type != 'maf':  # MAF can be slow for inverse operations
        print("\nDemonstrating invertibility...")
        with torch.no_grad():
            # Take a few samples
            test_samples = samples[:5]
            test_context = context[:5]
            
            # Encode to latents
            latents, _ = model.encode(test_samples, test_context)
            print(f"Latent shape: {latents.shape}")
            
            # Decode the latents
            reconstructed, _ = model.decode(latents, test_context)
            
            # Calculate reconstruction error
            mse = torch.mean((test_samples - reconstructed) ** 2, dim=1)
            print("Reconstruction MSE for 5 samples:")
            for i, err in enumerate(mse):
                print(f"Sample {i+1}: {err.item():.8f}")
                
            print(f"Average reconstruction MSE: {mse.mean().item():.8f}")

if __name__ == '__main__':
    main()
