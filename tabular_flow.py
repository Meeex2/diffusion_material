import torch
import torch.nn as nn
import normflows as nf

class TabularNormalizingFlow(nn.Module):
    """
    Normalizing flow with structured masking for tabular data.
    Designed to preserve column-wise dependencies using block coupling layers.
    """
    def __init__(self, dim=60, n_flows=8, hidden_dim=64):
        """
        Args:
            dim: Dimension of latent space (should be multiple of 4)
            n_flows: Number of flow layers
            hidden_dim: Hidden dimension for MLP networks
        """
        super().__init__()
        
        # Base distribution
        self.q0 = nf.distributions.DiagGaussian(dim)

        # Create flows
        flows = []
        
        # Block-wise binary mask for column-aware conditioning
        block_size = 4  # Each column is represented by 4 latent dimensions
        num_blocks = dim // block_size
        base_mask_pattern = torch.zeros(num_blocks)
        base_mask_pattern[::2] = 1  # Alternate blocks
        base_mask = base_mask_pattern.repeat_interleave(block_size).to(torch.bool)

        for i in range(n_flows):
            # Create scale and translation networks
            s = nf.nets.MLP([dim, hidden_dim, dim], init_zeros=True)
            t = nf.nets.MLP([dim, hidden_dim, dim], init_zeros=True)

            # Alternate block-wise masks
            mask = base_mask if i % 2 == 0 else ~base_mask
            
            # Add masked affine flow
            flows.append(nf.flows.MaskedAffineFlow(mask, t, s))
            
            # Add ActNorm for improved training stability
            flows.append(nf.flows.ActNorm(dim))
            
            # Add invertible linear mixing every 2 flows
            if i % 2 == 1 and i < n_flows - 1:
                flows.append(nf.flows.InvertibleLinear(dim))

        # Build the normalizing flow
        self.model = nf.NormalizingFlow(q0=self.q0, flows=flows)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for MLP components"""
        for module in self.modules():
            if isinstance(module, nf.nets.MLP):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Compute negative log-likelihood loss"""
        log_prob = self.model.log_prob(x)
        return -log_prob.mean()

    def sample(self, n_samples, device='cuda'):
        """Generate samples from the model"""
        z, _ = self.model.sample(num_samples=n_samples)
        return z

    def encode(self, x):
        """Map data to latent space with numerical stability checks"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        try:
            z = self.model.forward(x)
        except Exception as e:
            print(f"Error in encode: {str(e)}")
            raise
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        return z

    def decode(self, z):
        """Map latent space back to data space with stability checks"""
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        try:
            x, _ = self.model.inverse(z)
        except Exception as e:
            print(f"Error in decode: {str(e)}")
            raise
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x
