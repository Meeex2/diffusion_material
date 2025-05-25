import torch
import torch.nn as nn
import numpy as np
import normflows as nf

class TabularNormalizingFlow(nn.Module):
    """
    Normalizing flow model for tabular data with column-aware latent structure.
    Uses block-masked RealNVP with invertible linear mixing and increased capacity.
    """
    def __init__(self, dim, n_flows=8, hidden_dim=64):
        """
        Args:
            dim: Dimension of latent space (should be multiple of 4)
            n_flows: Number of flow layers
            hidden_dim: Hidden dimension for MLP networks (increased capacity)
        """
        super().__init__()
        # Create base distribution (standard normal)
        self.q0 = nf.distributions.DiagGaussian(dim)
        
        # Create flow layers
        flows = []
        
        # Block-wise binary mask for column-aware conditioning
        block_size = 4  # Each column represented by 4 latent dimensions
        num_blocks = dim // block_size
        mask_pattern = torch.zeros(num_blocks)
        mask_pattern[::2] = 1  # Alternate column blocks
        base_mask = mask_pattern.repeat_interleave(block_size).to(torch.bool)
        
        for i in range(n_flows):
            # Create scale and translation networks with increased capacity
            s = nf.nets.MLP([dim, hidden_dim, dim], 
                           init_zeros=True,
                           activation=nn.ReLU())
            t = nf.nets.MLP([dim, hidden_dim, dim], 
                           init_zeros=True,
                           activation=nn.ReLU())
            
            # Alternate block-wise masks
            mask = base_mask if i % 2 == 0 else ~base_mask
            
            # Add masked affine flow and stabilization layers
            flows.append(nf.flows.MaskedAffineFlow(mask, t, s))
            flows.append(nf.flows.ActNorm(dim))  # Per-dimension normalization
            
            # Add invertible linear mixing every 2 flows for better cross-block interaction
            if i % 2 == 1 and i < n_flows - 1:
                flows.append(nf.flows.InvertibleLinear(dim))
        
        # Create the normalizing flow model
        self.model = nf.NormalizingFlow(q0=self.q0, flows=flows)
        
        # Initialize all weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for MLP components"""
        for module in self.modules():
            if isinstance(module, nf.nets.MLP):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Compute negative log-likelihood loss"""
        log_prob = self.model.log_prob(x)
        return -log_prob.mean()

    def sample(self, n_samples, device='cuda'):
        """Generate samples from the model"""
        z, _ = self.model.sample(num_samples=n_samples)
        return z

    # Encoding/Decoding with numerical stability
    def encode(self, x):
        """Map data to latent space with stability checks"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        z = self.model.forward(x)
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        return z

    def decode(self, z):
        """Map latent space back to data space with stability checks"""
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        x, _ = self.model.inverse(z)
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x
