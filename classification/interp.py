from classification import config_classifier as config

def interpolate():

    ckpt = load_nerf2vec_checkpoint()

    encoder = Encoder(
                config.MLP_UNITS,
                config.ENCODER_HIDDEN_DIM,
                config.ENCODER_EMBEDDING_DIM
                )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=self.device)
        )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    # dset = NeRFDataset(dset_root, config.VAL_SPLIT)
    dset = os.path.abspath(os.path.join('data', 'validation.json'))  
    dset = NeRFDataset(val_dset_json, device='cpu')  
    
    while True:
        idx_A = randint(0, len(dset) - 1)
        train_nerf_A, test_nerf_A, matrices_unflattened_A, matrices_flattened_A, _, data_dir_A = dset[idx_A]
        
        """
        gt_rays_A = test_nerf_A['rays']
        gt_color_bkgds_A = test_nerf_A['color_bkgd']
        gt_rays_A = rays._replace(origins=rays_A.origins.cuda(), viewdirs=rays_A.viewdirs.cuda())
        gt_color_bkgds_A = color_bkgds_A.cuda()
        """
        



        class_id_B = -1
        while class_id_B != class_id_A:
            idx_B = randint(0, len(dset) - 1)
            train_nerf_B, test_nerf_B, matrices_unflattened_B, matrices_flattened_B, _, data_dir_B = dset[idx_B]
        
        matrices_A = matrix_A.cuda()
        matrices_B = matrix_B.cuda()

        

        with torch.no_grad():
            embedding_A = encoder(matrices_flattened_A)  # TODO: check dimensions
            embedding_B = encoder(matrices_flattened_B)  # TODO: check dimensions
        

        rgb_A, _, _, _ = render_image(
                    radiance_field=decoder,
                    embeddings=embedding_A,
                    occupancy_grid=None,
                    rays=rays,
                    scene_aabb=self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds.unsqueeze(dim=0),
                    grid_weights=None
                )

        
        embeddings = [embedding_A]
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            emb_interp = (1 - gamma) * embedding_A + gamma * embedding_B
            embeddings.append(emb_interp)
        embeddings.append(embedding_B)


        rays = test_nerf_A['rays']
        color_bkgds = test_nerf_A['color_bkgd']
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
        color_bkgds = color_bkgds.cuda()

        renderings = []

        for i in range(len(embeddings)):
            rgb, _, _, _ = render_image(
                    radiance_field=decoder,
                    embeddings=embeddings[idx].unsqueeze(dim=0),
                    occupancy_grid=None,
                    rays=rays,
                    scene_aabb=self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds.unsqueeze(dim=0),
                    grid_weights=curr_grid_weights
                )
            
            renderings.append(rgb)


        pred_image = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
        gt_image_A = wandb.Image((pixels_A.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))
        gt_image_B = wandb.Image((pixels_B.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))

        self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image]})