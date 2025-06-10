
#### 1. New Model Architecture (model.py)
```python
class InvertibleMLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        
        # Use invertible blocks for better reconstruction
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_t, dim_t * 2),
                nn.SiLU(),
                nn.Linear(dim_t * 2, dim_t),
                nn.SiLU()
            ) for _ in range(4)
        ])
        
        self.time_embed = nn.Sequential(
            PositionalEmbedding(num_channels=dim_t),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        # Final layer with residual connection
        self.final = nn.Linear(dim_t, d_in)
    
    def forward(self, x, t, reverse=False):
        # Time embedding
        emb = self.time_embed(t)
        
        # Main processing
        h = self.proj(x) + emb
        
        if not reverse:
            for block in self.blocks:
                h = h + block(h)
        else:
            # For inversion, process in reverse order
            for block in reversed(self.blocks):
                # Approximate inverse with residual
                h = h - block(h)
        
        return self.final(h)


class InvertiblePrecond(nn.Module):
    def __init__(self, denoise_fn, hid_dim, sigma_min=0.002, sigma_max=80, sigma_data=0.5):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.register_buffer('alphas', torch.linspace(1, 0, 1000))
    
    def forward(self, x, sigma, reverse=False):
        # Convert sigma to alpha
        alpha = self.sigma_to_alpha(sigma)
        
        # Denoising step
        denoised = self.denoise_fn(x, sigma, reverse)
        
        if not reverse:
            # Forward diffusion: x = alpha * x0 + (1-alpha) * noise
            noise = torch.randn_like(x)
            return alpha.sqrt() * denoised + (1 - alpha).sqrt() * noise
        else:
            # Reverse diffusion: x0 = (x - (1-alpha).sqrt() * noise)/alpha.sqrt()
            noise = (x - alpha.sqrt() * denoised) / (1 - alpha).sqrt().clamp(min=1e-8)
            return denoised + noise
    
    def sigma_to_alpha(self, sigma):
        # Map sigma to alpha in [0,1]
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        return 1 / (1 + sigma**2)


class InvertibleModel(nn.Module):
    def __init__(self, denoise_fn, hid_dim, num_timesteps=1000):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.precond = InvertiblePrecond(denoise_fn, hid_dim)
        self.num_timesteps = num_timesteps
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, reverse=False):
        if not reverse:
            # Training mode
            t = torch.randint(0, self.num_timesteps, (x.size(0), device=x.device)
            sigma = self.t_to_sigma(t)
            noisy_x = self.precond(x, sigma)
            denoised = self.denoise_fn(noisy_x, sigma)
            return self.loss_fn(denoised, x)
        else:
            # Inversion mode
            return self.invert(x)
    
    def t_to_sigma(self, t):
        # Map timestep to sigma
        return self.precond.sigma_max * (self.precond.sigma_min / self.precond.sigma_max) ** (t.float() / (self.num_timesteps - 1))
    
    def invert(self, x, steps=50):
        # Invert the diffusion process
        alphas = self.precond.alphas
        x_prev = x.clone()
        
        for i in range(steps - 1, -1, -1):
            t = torch.full((x.size(0), i, device=x.device)
            sigma = self.t_to_sigma(t)
            x_prev = self.precond(x_prev, sigma, reverse=True)
        
        return x_prev
```

#### 2. Training Code (train.py)
```python
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
from tabsyn.model import InvertibleMLPDiffusion, InvertibleModel  # Use new model classes

warnings.filterwarnings('ignore')

def main(args): 
    device = args.device
    train_z, _, _, ckpt_path, _ = get_input_train(args)
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    num_epochs = 10000 + 1
    num_timesteps = 1000  # DDIM-like timesteps

    denoise_fn = InvertibleMLPDiffusion(in_dim, 1024).to(device)
    model = InvertibleModel(denoise_fn, train_z.shape[1], num_timesteps).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()
    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            
            # Forward pass with diffusion
            loss = model(inputs)
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Training Time: ', end_time - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabSyn')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    main(args)
```

#### 3. Sampling and Inversion Code (sample.py)
```python
import torch
import argparse
import warnings
import time
import os
import numpy as np
import pandas as pd
from tabsyn.model import InvertibleMLPDiffusion, InvertibleModel
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target

warnings.filterwarnings('ignore')

def ddim_sample(model, num_samples, dim, steps=50, device='cuda:0'):
    """Deterministic DDIM sampling"""
    # Start from pure noise
    x = torch.randn(num_samples, dim, device=device)
    
    # Timesteps from noise to data
    timesteps = torch.linspace(0, model.num_timesteps - 1, steps, device=device).long()
    
    for t in reversed(timesteps):
        sigma = model.t_to_sigma(t)
        denoised = model.denoise_fn(x, sigma)
        alpha = model.precond.sigma_to_alpha(sigma)
        
        # DDIM update
        if t > 0:
            prev_sigma = model.t_to_sigma(t - 1)
            prev_alpha = model.precond.sigma_to_alpha(prev_sigma)
            noise = (x - alpha.sqrt() * denoised) / (1 - alpha).sqrt()
            x = prev_alpha.sqrt() * denoised + (1 - prev_alpha).sqrt() * noise
        else:
            x = denoised
    
    return x

def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps or 50
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 
    mean = train_z.mean(0)

    denoise_fn = InvertibleMLPDiffusion(in_dim, 1024).to(device)
    model = InvertibleModel(denoise_fn, train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    model.eval()

    # Generating samples
    start_time = time.time()
    num_samples = train_z.shape[0]
    sample_dim = in_dim

    # Generate samples
    x_next = ddim_sample(model, num_samples, sample_dim, steps, device)
    
    # Save generated samples
    x_next = x_next * 2 + mean.to(device)
    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    
    # Apply column name mapping
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)
    
    # Latent retrieval demonstration
    sample_idx = 0
    sample_tensor = torch.tensor(syn_data[sample_idx], device=device).unsqueeze(0).float()
    
    # Retrieve latent through inversion
    recovered_latent = model.invert(sample_tensor, steps=steps)
    
    # Reconstruct sample from recovered latent
    reconstructed_sample = ddim_sample(model, 1, sample_dim, steps, device, recovered_latent)
    
    # Post-processing
    reconstructed_sample = reconstructed_sample * 2 + mean.to(device)
    reconstructed_data = reconstructed_sample.float().cpu().numpy()
    
    # Calculate reconstruction error
    reconstruction_error = np.abs(syn_data[sample_idx] - reconstructed_data[0]).mean()
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Saving sampled data to', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save generated data')
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    main(args)
```
