import os
import sys

from modules import *
from train import *


style_gan = load_model()

def predict(path):
    weights_path = keras.utils.get_file(
        "predict",
        path,
        extract=True,
        cache_dir=os.path.abspath("."),
        cache_subdir="predicted",
    )

    style_gan.grow_model(128)
    style_gan.load_weights(os.path.join("predicted/stylegan"))

    tf.random.set_seed(196)
    batch_size = 2
    norm = tf.random.normal((batch_size, style_gan.z_dim))
    w = style_gan.mapping(norm)
    noise = style_gan.generate_noise(batch_size=batch_size)
    all_images = style_gan({"style_code": w, "noise": noise, "alpha": 1.0})
    plot_images(all_images, 5)


if __name__ == "__main__":
    predict(sys.argv[1])
