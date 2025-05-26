import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
import numpy as np
from tabsyn.latent_utils import get_input_train
from tabsyn.tabular_normalizing_flow import ConditionalTabularFlow, extract_correlation_context

warnings.filterwarnings('ignore')

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

def correlation_loss(real_corr, fake_corr):
    """
    Compute loss between real and fake correlation matrices
    """
    return torch.nn.functional.mse_loss(real_corr, fake_corr)

def main(args): 
    device = args.device

    train_z, _, _, ckpt_path, _ = get_input_train(args)
    
    # Ensure directory exists
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]
    n_columns = args.n_columns
    
    # Verify column count
    if in_dim % n_columns != 0:
        print(f"Warning: input dimension {in_dim} is not divisible by column count {n_columns}")
        print(f"Adjusting column count to {in_dim // (in_dim // n_columns)}")
        n_columns = in_dim // (in_dim // n_columns)
    
    # Compute global correlation for the dataset
    global_corr = compute_batch_correlation(train_z, n_columns)
    print(f"Global correlation matrix shape: {global_corr.shape}")
    
    # Normalize data properly
    mean, std = train_z.mean(0), train_z.std(0)
    train_z_norm = (train_z - mean) / std  # Use standard normalization
    
    # Create context tensors (correlation matrices)
    batch_size = args.batch_size
    
    # Create custom dataset to store both data and contexts
    train_contexts = extract_correlation_context(train_z, n_columns)
    train_dataset = TensorDataset(train_z_norm, train_contexts)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = args.max_epochs + 1

    # Create the conditional normalizing flow model
    model = ConditionalTabularFlow(
        dim=in_dim, 
        n_flows=args.n_flows,
        hidden_dim=args.hidden_dim,
        n_columns=n_columns,
        flow_type=args.flow_type
    ).to(device)
    
    # Convert to double precision for better numerical stability
    model = model.double()
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Initialize optimizer with learning rate from args
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Create a model name with hyperparameters
    model_name = f"flow_model_type{args.flow_type}_nflows{args.n_flows}_hdim{args.hidden_dim}_lr{args.learning_rate}_lambda{args.lambda_corr}"
    
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        batch_nll_loss = 0.0
        batch_corr_loss = 0.0
        len_input = 0
        
        for batch_data, batch_context in pbar:
            inputs = batch_data.double().to(device)
            contexts = batch_context.double().to(device)
            
            # Forward pass
            nll_loss = model(inputs, contexts)
            nll_loss = nll_loss.mean()
            
            # Generate samples for correlation loss if lambda_corr > 0
            corr_loss = torch.tensor(0.0, device=device)
            if args.lambda_corr > 0:
                with torch.no_grad():
                    # Sample using the same contexts
                    samples = model.sample(len(inputs), contexts)
                    
                    # Compute correlation matrices
                    real_corr = compute_batch_correlation(inputs, n_columns)
                    fake_corr = compute_batch_correlation(samples, n_columns)
                    
                    # Compute correlation loss
                    corr_loss = correlation_loss(real_corr, fake_corr)
            
            # Total loss
            loss = nll_loss + args.lambda_corr * corr_loss

            batch_loss += loss.item() * len(inputs)
            batch_nll_loss += nll_loss.item() * len(inputs)
            batch_corr_loss += corr_loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item(), "NLL": nll_loss.item(), "Corr": corr_loss.item()})

        # Calculate average losses
        curr_loss = batch_loss / len_input
        avg_nll_loss = batch_nll_loss / len_input
        avg_corr_loss = batch_corr_loss / len_input
        
        print(f"Epoch {epoch+1} - Avg Loss: {curr_loss:.6f}, NLL: {avg_nll_loss:.6f}, Corr: {avg_corr_loss:.6f}")
        
        scheduler.step(curr_loss)

        # Save checkpoint if best model
        if curr_loss < best_loss:
            best_loss = curr_loss
            patience_counter = 0
            
            # Save with more information in the checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'nll_loss': avg_nll_loss,
                'corr_loss': avg_corr_loss,
                'lambda_corr': args.lambda_corr,
                'flow_type': args.flow_type,
                'n_flows': args.n_flows,
                'hidden_dim': args.hidden_dim,
                'n_columns': n_columns,
                'mean': mean,
                'std': std,
                'global_corr': global_corr
            }
            torch.save(checkpoint, f'{ckpt_path}/{model_name}.pt')
            print(f"Saved best model with loss {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter == args.patience:
                print('Early stopping')
                break

        # Periodic checkpoint saving
        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_loss,
                'lambda_corr': args.lambda_corr,
                'flow_type': args.flow_type
            }, f'{ckpt_path}/{model_name}_{epoch}.pt')

    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Conditional Normalizing Flow')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--n_flows', type=int, default=8, help='Number of flow layers.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size.')
    parser.add_argument('--n_columns', type=int, default=15, help='Number of columns in the original data.')
    parser.add_argument('--flow_type', type=str, default='nsf_ar', 
                        choices=['nsf_ar', 'nsf_c', 'maf', 'realnvp'], 
                        help='Type of flow to use.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lambda_corr', type=float, default=0.1, 
                        help='Weight for correlation loss (0 to disable).')
    parser.add_argument('--max_epochs', type=int, default=10000, help='Maximum number of epochs.')
    parser.add_argument('--patience', type=int, default=500, help='Patience for early stopping.')
    
    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
