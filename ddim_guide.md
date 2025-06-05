
```
def ddim_sample(net, num_samples, sample_dim, num_steps=50, device='cuda:0'):
    # Parameters from the original training setup
    sigma_min = max(0.002, net.sigma_min)  # Lower bound of noise
    sigma_max = min(80, net.sigma_max)      # Upper bound of noise
    rho = 7                                # Time step exponent
    sigma_data = 0.5                       # As defined in Precond

    # Create timesteps (same schedule as training)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Add final zero step

    # Initialize latents (start from random noise)
    latents = torch.randn([num_samples, sample_dim], device=device)
    x_next = latents * t_steps[0]  # Scale by initial sigma

    # DDIM deterministic sampling loop
    with torch.no_grad():
        for i in range(num_steps):
            current_sigma = t_steps[i]
            next_sigma = t_steps[i+1]
            
            # Convert sigma to alpha (VP schedule)
            current_alpha = 1 / (1 + current_sigma**2)
            next_alpha = 1 / (1 + next_sigma**2)

            # Get model prediction (denoised sample)
            sigma_tensor = current_sigma * torch.ones(num_samples, device=device)
            pred_x0 = net(x_next, sigma_tensor)  # Model predicts denoised data

            # DDIM update rule (deterministic)
            term1 = torch.sqrt(next_alpha) * pred_x0
            term2 = torch.sqrt(1 - next_alpha) * (x_next - torch.sqrt(current_alpha) * pred_x0) 
            term2 /= torch.sqrt(1 - current_alpha)
            
            x_next = term1 + term2

    return x_next
```

```
# Original sampling call (replace this)
# x_next = sample(model.denoise_fn_D, num_samples, sample_dim)

# New deterministic DDIM sampling
x_next = ddim_sample(
    model.denoise_fn_D, 
    num_samples, 
    sample_dim,
    num_steps=args.steps or 50,  # Use steps from args
    device=device
)
```

More sophisticated adaptation

### Implementation Steps:

#### 1. Update the Model Architecture

```python
class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        
        # Use timestep embedding instead of noise labels
        self.time_embed = nn.Sequential(
            PositionalEmbedding(num_channels=dim_t),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, t, class_labels=None):
        # t is now timestep index
        emb = self.time_embed(t)
        x = self.proj(x) + emb
        return self.mlp(x)
```

#### 2. Create a DDIM-Compatible Scheduler


```python
class EDM2DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, sigma_min=0.002, sigma_max=80, rho=7):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        # Create sigma schedule (same as your original EDM setup)
        step_indices = torch.arange(num_train_timesteps)
        self.sigmas = (sigma_max ** (1 / rho) + step_indices / (num_train_timesteps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])
        
        # Create alpha schedule for DDIM
        self.alphas = 1 / (1 + self.sigmas[:-1] ** 2)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0)
        self.timesteps = timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
    
    def step(self, model_output, sample, timestep_idx, eta=0.0):
        # Get current and previous timesteps
        t = self.timesteps[timestep_idx]
        prev_t = self.timesteps[timestep_idx + 1] if timestep_idx < len(self.timesteps) - 1 else -1
        
        # Get corresponding sigma values
        sigma = self.sigmas[t]
        prev_sigma = self.sigmas[prev_t] if prev_t >= 0 else 0
        
        # Compute alpha and sigma values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        
        # Predicted original sample (x0)
        pred_x0 = (sample - model_output * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prod_t_prev - eta ** 2) * model_output
        
        # Update sample
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt
        
        return prev_sample, pred_x0
```

#### 3. Implement DDIM Sampling Function

```python
def ddim_sample(model, num_samples, sample_dim, num_steps=50, device='cuda'):
    scheduler = EDM2DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_steps, device=device)
    
    # Initial noise
    x = torch.randn([num_samples, sample_dim], device=device) * scheduler.sigmas[0]
    
    # Sampling loop
    for i, t_idx in enumerate(scheduler.timesteps):
        # Get model output
        model_output = model(x, t_idx)
        
        # DDIM step
        x, _ = scheduler.step(
            model_output=model_output,
            sample=x,
            timestep_idx=i,
            eta=0.0  # Fully deterministic
        )
    
    return x
```

#### 4. Update the Generation Code
Modify your generation script to use the new DDIM sampling:

```python
def main(args):
    # ... [previous setup code] ...
    
    # Generating samples
    start_time = time.time()
    num_samples = train_z.shape[0]
    sample_dim = in_dim

    # Use DDIM sampling
    x_next = ddim_sample(
        model.denoise_fn_D, 
        num_samples, 
        sample_dim,
        num_steps=args.steps or 50,
        device=device
    )
    
    # Post-processing remains the same
    x_next = x_next * 2 + mean.to(device)
    # ... [rest of the code] ...
```

