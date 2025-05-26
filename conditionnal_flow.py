import torch
import torch.nn as nn
import numpy as np
import normflows as nf

class ConditionalTabularFlow(nn.Module):
    """
    Conditional normalizing flow model for tabular data.
    Conditions on correlation matrix to preserve relationships between columns.
    """
    def __init__(self, dim, n_flows=8, hidden_dim=128, n_columns=15, flow_type='nsf_ar'):
        """
        Args:
            dim: Dimension of the data/latent space
            n_flows: Number of flow layers
            hidden_dim: Hidden dimension for the MLP networks
            n_columns: Number of columns in the original tabular data
            flow_type: Type of flow ('nsf_ar', 'nsf_c', 'maf', or 'realnvp')
        """
        super().__init__()
        
        # Calculate context size - upper triangular part of correlation matrix (excluding diagonal)
        self.n_columns = n_columns
        self.context_size = n_columns * (n_columns - 1) // 2
        
        # Create base distribution (standard normal)
        self.q0 = nf.distributions.DiagGaussian(dim)
        
        # Create flow layers
        flows = []
        
        # Choose flow type based on parameter
        if flow_type == 'nsf_ar':
            # Neural Spline Flow (Autoregressive)
            for i in range(n_flows):
                flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
                    dim, 2, hidden_dim, num_context_channels=self.context_size))
                flows.append(nf.flows.ActNorm(dim))
                if i < n_flows - 1:  # No permutation after last layer
                    flows.append(nf.flows.LULinearPermute(dim))
        
        elif flow_type == 'nsf_c':
            # Neural Spline Flow (Coupling)
            for i in range(n_flows):
                # Create binary mask for coupling layers (column-aware)
                mask = self._create_column_aware_mask(dim, n_columns, i)
                
                flows.append(nf.flows.CoupledRationalQuadraticSpline(
                    dim, 2, hidden_dim, mask=mask, num_context_channels=self.context_size))
                flows.append(nf.flows.ActNorm(dim))
                if i < n_flows - 1:  # No permutation after last layer
                    flows.append(nf.flows.LULinearPermute(dim))
        
        elif flow_type == 'maf':
            # Masked Autoregressive Flow
            for i in range(n_flows):
                flows.append(nf.flows.MaskedAffineAutoregressive(
                    dim, hidden_dim, context_features=self.context_size, num_blocks=2))
                flows.append(nf.flows.ActNorm(dim))
                if i < n_flows - 1:  # No permutation after last layer
                    flows.append(nf.flows.LULinearPermute(dim))
        
        else:  # Default to RealNVP
            # Create binary mask for coupling layers (column-aware)
            for i in range(n_flows):
                # Create scale and translation networks with context
                s = self._create_context_net(dim, hidden_dim, self.context_size)
                t = self._create_context_net(dim, hidden_dim, self.context_size)
                
                # Create column-aware mask
                mask = self._create_column_aware_mask(dim, n_columns, i)
                
                flows.append(nf.flows.MaskedAffineFlow(mask, t, s))
                flows.append(nf.flows.ActNorm(dim))
        
        # Create the conditional normalizing flow model
        self.model = nf.ConditionalNormalizingFlowModel(self.q0, flows)
        
    def _create_column_aware_mask(self, dim, n_columns, layer_idx):
        """
        Create masks that respect column boundaries for better correlation preservation
        """
        features_per_column = dim // n_columns
        mask = torch.zeros(dim)
        
        if layer_idx % 2 == 0:
            # Mask odd columns
            for col in range(1, n_columns, 2):
                start_idx = col * features_per_column
                end_idx = start_idx + features_per_column
                mask[start_idx:end_idx] = 1
        else:
            # Mask even columns
            for col in range(0, n_columns, 2):
                start_idx = col * features_per_column
                end_idx = start_idx + features_per_column
                mask[start_idx:end_idx] = 1
                
        return mask
    
    def _create_context_net(self, dim, hidden_dim, context_dim):
        """Create a conditional network that takes context as input"""
        return nf.nets.ContextMLP([dim, hidden_dim, hidden_dim, dim], 
                                  context_dim, 
                                  init_zeros=True)
    
    def forward(self, x, context):
        """
        Compute negative log-likelihood loss
        Args:
            x: Input data tensor [batch_size, dim]
            context: Correlation context [batch_size, context_size]
        Returns:
            Negative log-likelihood loss
        """
        log_prob = self.model.log_prob(x, context=context)
        return -log_prob
    
    def sample(self, n_samples, context, device='cuda'):
        """
        Generate samples from the model conditioned on correlation context
        Args:
            n_samples: Number of samples to generate
            context: Correlation context [batch_size, context_size]
            device: Device to generate samples on
        Returns:
            Samples tensor [n_samples, dim]
        """
        z, _ = self.model.sample(n_samples, context=context)
        return z
    
    def encode(self, x, context):
        """
        Encode data to latent space with numerical stability checks
        """
        # Check for NaN or Inf values in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        z, log_det = self.model.forward(x, context=context)
        
        # Check for NaN or Inf values in output
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"Warning: Encoded latent contains NaN or Inf values")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return z, log_det

    def decode(self, z, context):
        """
        Decode from latent space to data space with numerical stability checks
        """
        # Check for NaN or Inf values in input
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"Warning: Latent contains NaN or Inf values")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x, log_det = self.model.inverse(z, context=context)
        
        # Check for NaN or Inf values in output
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Decoded output contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x, log_det

def extract_correlation_context(data, n_columns=15):
    """
    Extract the correlation matrix from data and flatten the upper triangular part.
    Args:
        data: Input data tensor [batch_size, dim]
        n_columns: Number of columns in the original tabular data
    Returns:
        Flattened correlation matrix (upper triangular part) [batch_size, n_columns*(n_columns-1)/2]
    """
    batch_size = data.shape[0]
    features_per_column = data.shape[1] // n_columns
    
    # Reshape data to get column-wise representation
    reshaped_data = data.view(batch_size, n_columns, features_per_column)
    
    # Get column means (averaging over the feature dimension)
    column_means = reshaped_data.mean(dim=2)
    
    # Calculate correlation matrix for each item in batch
    corr_matrices = []
    for i in range(batch_size):
        # Calculate correlation matrix
        corr = torch.corrcoef(column_means[i].unsqueeze(0))[0]
        
        # Extract upper triangular part (excluding diagonal)
        triu_indices = torch.triu_indices(n_columns, n_columns, offset=1)
        flat_corr = corr[triu_indices[0], triu_indices[1]]
        
        corr_matrices.append(flat_corr)
    
    # Stack correlation matrices
    return torch.stack(corr_matrices)
