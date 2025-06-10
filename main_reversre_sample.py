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
    
    '''
        Generating samples    
    '''
    start_time = time.time()
    num_samples = train_z.shape[0]
    sample_dim = in_dim

    # Generate samples and get initial latents
    x_next, init_latents = deterministic_sample(
        model.denoise_fn_D, 
        num_samples, 
        sample_dim,
        num_steps=steps or 50,
        device=device
    )
    
    # Save initial latents for later use
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
    
    '''
        Latent retrieval demonstration
    '''
    # Select a sample to demonstrate latent retrieval
    sample_idx = 0
    sample_tensor = torch.tensor(syn_data[sample_idx], device=device).unsqueeze(0)
    
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


def retrieve_latent(net, sample, num_steps=50, device='cuda:0'):
    """Recover initial latent from a generated sample"""
    # Create the same sigma schedule as forward process
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    
    sigma_min = max(0.002, net.sigma_min)
    sigma_max = min(80, net.sigma_max)
    rho = 7
    
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # Reverse the time steps
    rev_t_steps = t_steps.flip(0)
    x_prev = sample.clone()
    
    # Reverse diffusion process
    with torch.no_grad():
        for i, (t_prev, t_cur) in enumerate(zip(rev_t_steps[:-1], rev_t_steps[1:])):
            # Compute denoised version at current step
            denoised = net(x_prev, t_prev).to(torch.float32)
            
            # Reverse Heun's method correction
            if i > 0:  # Apply correction for all but first step
                # First get the Euler step estimate
                d_cur = (x_prev - denoised) / t_prev
                x_euler = x_prev - (t_cur - t_prev) * d_cur
                
                # Get the corrected gradient
                denoised2 = net(x_euler, t_cur).to(torch.float32)
                d_prime = (x_euler - denoised2) / t_cur
                
                # Compute the reverse step
                x_prev = x_prev - (t_cur - t_prev) * (0.5 * d_cur + 0.5 * d_prime)
            else:
                # For the first step (t=0), use simple Euler reverse
                d_cur = (x_prev - denoised) / t_prev
                x_prev = x_prev - (t_cur - t_prev) * d_cur
    
    # The final state is the initial latent
    return x_prev
