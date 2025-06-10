def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 
    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    
    # Generating samples
    num_samples = train_z.shape[0]
    sample_dim = in_dim
    start_time = time.time()
    
    # Generate samples and get initial latents
    x_next, init_latents = deterministic_sample(
        model.denoise_fn_D, 
        num_samples, 
        sample_dim,
        num_steps=steps or 50,
        device=device
    )
    
    # Save initial latents
    latent_path = os.path.join(os.path.dirname(save_path), f"{os.path.basename(save_path)}_latents.pt")
    torch.save(init_latents.cpu(), latent_path)
    
    # Post-processing
    x_next = x_next * 2 + mean.to(device)
    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    
    # Apply column name mapping
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)
    
    # Latent retrieval demonstration
    sample_idx = 0  # First sample
    sample_tensor = torch.tensor(syn_data[sample_idx], device=device).unsqueeze(0).float()
    
    # Retrieve latent from the sample
    recovered_latent = retrieve_latent(
        model.denoise_fn_D,
        sample_tensor,
        num_steps=steps or 50,
        device=device
    )
    
    # Reconstruct sample from recovered latent
    reconstructed_sample, _ = deterministic_sample(
        model.denoise_fn_D,
        num_samples=1,
        dim=sample_dim,
        num_steps=steps or 50,
        device=device,
        latent=recovered_latent
    )
    
    # Post-process reconstructed sample
    reconstructed_sample = reconstructed_sample * 2 + mean.to(device)
    reconstructed_data = reconstructed_sample.float().cpu().numpy()
    
    # Calculate reconstruction error
    reconstruction_error = np.abs(syn_data[sample_idx] - reconstructed_data[0]).mean()
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Saving sampled data to', save_path)
    print('Saving initial latents to', latent_path)

###########

def deterministic_sample(net, num_samples, dim, num_steps=50, device='cuda:0', latent=None):
    """Deterministic sampling with optional latent input and latent return"""
    if latent is None:
        # Create new initial latent
        latent = torch.randn([num_samples, dim], device=device)
    
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    sigma_min = max(0.002, net.sigma_min)
    sigma_max = min(80, net.sigma_max)
    rho = 7

    # Create sigma schedule
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Initialize state
    x_cur = latent.to(torch.float32) * t_steps[0]
    x_next = x_cur

    # Sampling loop
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Update current state
            x_cur = x_next
            
            # Remove stochastic elements (gamma=0, S_noise=0)
            gamma = 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur  # No noise added
            
            # First Euler step
            denoised = net(x_hat, t_hat).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction (Heun's method)
            if i < num_steps - 1:
                denoised2 = net(x_next, t_next).to(torch.float32)
                d_prime = (x_next - denoised2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next, latent


def retrieve_latent(net, sample, num_steps=50, device='cuda:0', eps=1e-9):
    """Recover initial latent from a generated sample with robust handling of t=0"""
    # Create the same sigma schedule as forward process
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    
    sigma_min = max(0.002, net.sigma_min)
    sigma_max = min(80, net.sigma_max)
    rho = 7
    
    # Create forward sigma schedule
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # Reverse the time steps, skipping t=0 initially
    rev_t_steps = t_steps.flip(0)[1:]  # Start from last non-zero sigma
    x_prev = sample.clone()
    
    # Add small epsilon to prevent division by zero
    safe_t_steps = t_steps.clone()
    safe_t_steps[-1] = eps  # Replace 0 with epsilon
    
    # Reverse diffusion process
    with torch.no_grad():
        # First handle t=0 separately
        t_prev = safe_t_steps[-1]  # Use epsilon instead of 0
        denoised = net(x_prev, t_prev).to(torch.float32)
        d_cur = (x_prev - denoised) / t_prev
        
        # Process remaining steps in reverse order
        for i, t_cur in enumerate(rev_t_steps):
            # Calculate time step difference
            dt = t_cur - t_prev
            
            # Get denoised estimate at current position
            denoised_cur = net(x_prev, t_prev).to(torch.float32)
            
            # Compute gradient
            d_cur = (x_prev - denoised_cur) / t_prev
            
            # Apply reverse Euler step
            x_next = x_prev - dt * d_cur
            
            # Apply Heun's correction if not last step
            if i < len(rev_t_steps) - 1:
                # Estimate at next position
                denoised_next = net(x_next, t_cur).to(torch.float32)
                d_prime = (x_next - denoised_next) / t_cur
                
                # Average gradients
                avg_d = 0.5 * (d_cur + d_prime)
                
                # Recompute position with better gradient estimate
                x_next = x_prev - dt * avg_d
            
            # Update for next iteration
            x_prev = x_next
            t_prev = t_cur
    
    # The final state is the initial latent
    return x_prev
