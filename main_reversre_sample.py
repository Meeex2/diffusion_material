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
