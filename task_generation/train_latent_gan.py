"""
This module, `train_latent_gan.py`, is used to train a Generative Adversarial Network (GAN) on the latent codes of a given class. 
The GAN consists of a generator and a discriminator, both with two layers, which are imported from the `latent_3d_points.src.generators_discriminators` module. 
The training data is derived from the embeddings obtained by nerf2vec. 
The output of the training process is stored in a directory specified by `paths.GENERATION_OUT_DIR`, with the directory name formatted to include the class index.
"""

### To be used with the code from the repository https://github.com/optas/latent_3d_points

import os.path as osp

import numpy as np
from latent_3d_points.src.generators_discriminators import (
    latent_code_discriminator_two_layers,
    latent_code_generator_two_layers,
)
from latent_3d_points.src.in_out import PointCloudDataSet, create_dir
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.w_gan_gp import W_GAN_GP

import paths


def train(class_idx=0):

    experiment_name = 'nerf2vec_{}'.format(class_idx)
    top_out_dir = paths.GENERATION_OUT_DIR.format(experiment_name)
    embedding_size = 1024
    n_epochs = 2000
    n_syn_samples = 1000  # how many synthetic samples to produce at each save step
    saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 50)])

    latent_codes = np.load("{}_{}.npz".format(paths.GENERATION_EMBEDDING_DIR, class_idx))["embeddings"]
    latent_data = PointCloudDataSet(latent_codes)
    print(latent_data.num_examples)

    # optimization parameters
    init_lr = 0.0001
    batch_size = 50
    noise_params = {"mu": 0, "sigma": 0.2}
    beta = 0.5  # ADAM's momentum

    train_dir = osp.join(top_out_dir, "latent_gan_ckpts")  # TODO: test this path
    create_dir(train_dir)
    synthetic_data_out_dir = osp.join(top_out_dir, "generated_embeddings") # TODO: test this path
    create_dir(synthetic_data_out_dir)

    reset_tf_graph()

    gan = W_GAN_GP(
        experiment_name,
        init_lr,
        10,
        [embedding_size],
        embedding_size,
        latent_code_discriminator_two_layers,
        latent_code_generator_two_layers,
        beta=beta,
    )

    print("Start")

    for _ in range(n_epochs):
        loss, duration = gan._single_epoch_train(latent_data, batch_size, noise_params)
        epoch = int(gan.sess.run(gan.increment_epoch))
        print("epoch:", epoch, "loss:", loss)

        if epoch in saver_step:
            checkpoint_path = osp.join(train_dir, "epoch_" + str(epoch) + ".ckpt")
            gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

            syn_latent_data = gan.generate(n_syn_samples, noise_params)
            np.savez(
                osp.join(synthetic_data_out_dir, "epoch_" + str(epoch) + ".npz"),
                embeddings=syn_latent_data,
            )