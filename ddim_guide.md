
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
