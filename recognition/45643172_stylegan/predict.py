import os
import sys

from modules import *
from train import *



def predict(url, name):
    weights_path = keras.utils.get_file(
        name,
        url,
        extract=True,
        cache_dir=os.path.abspath("."),
        cache_subdir="pretrained",
    )

    style_gan.grow_model(128)
    style_gan.load_weights(os.path.join("pretrained/stylegan"))

    tf.random.set_seed(196)
    batch_size = 2
    z = tf.random.normal((batch_size, style_gan.z_dim))
    w = style_gan.mapping(z)
    noise = style_gan.generate_noise(batch_size=batch_size)
    images = style_gan({"style_code": w, "noise": noise, "alpha": 1.0})
    plot_images(images, 5)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
