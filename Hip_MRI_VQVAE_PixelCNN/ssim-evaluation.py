import tensorflow as tf

import importlib
vqvae = importlib.import_module("vq-vae")

"""
EVALUATED: 0.78039
"""

def evaluate_ssim(model, test_dataset, max_val=1.0):
    """
    Compute average SSIM over a test dataset for a VQ-VAE model.

    Args:
        model: a trained VQ-VAE model with .predict or __call__ method
        test_dataset: a tf.data.Dataset yielding batches of images in [0, 1] range
        max_val: maximum pixel value (default assumes images scaled to [0,1])

    Returns:
        float: average SSIM score across the dataset
    """
    ssim_scores = []

    for batch in test_dataset:
        # Ensure batch is float32
        batch = tf.cast(batch, tf.float32)

        # Get reconstructions from model
        recon = model(batch, training=False)

        # Compute SSIM for each image in the batch
        batch_ssim = tf.image.ssim(batch, recon, max_val=max_val)

        # Collect as numpy values
        ssim_scores.extend(batch_ssim.numpy())

    return float(tf.reduce_mean(ssim_scores).numpy())

if __name__ == "__main__":
    # fetch test set
    test = vqvae.get_dataset(
        # "data/keras_slices_data/keras_slices_train/"
        "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
    ).batch(10)

    vae = vqvae.load_model(
        "trainer.weights.h5"
    )

    ssim = evaluate_ssim(
        vae, test
    )

    print(ssim)