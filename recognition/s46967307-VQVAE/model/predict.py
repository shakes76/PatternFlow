import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
import numpy as np
import matplotlib.pyplot as plt
from modules import AE, get_pixel_cnn, latent_dims, num_embeddings, get_indices, pixelcnn_input_shape, image_shape, encoded_image_shape
from dataset import load_data

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)

model: AE = tf.keras.models.load_model("model.ckpt")
print(model.summary())

data = load_data()

def visualize_autoencoder(n):
    predictions = model.predict(tf.reshape(data["test"][0:30], shape=(-1,*image_shape[0:2])))
    encoded = get_indices(model.vq.variables[0], tf.reshape(model.encoder.predict(data["test"]), shape=(-1,latent_dims)), quantize=False)
    encoded = tf.reshape(encoded, shape=(-1, *pixelcnn_input_shape[0:2]))
    plt.tight_layout()
    fig, axs = plt.subplots(n, 3, figsize=image_shape[0:2])
    for i in range(n):
        axs[i,0].imshow(tf.reshape(data["test"][i], shape=image_shape[0:2]))
        axs[i,2].imshow(predictions[i])
        axs[i,1].imshow(tf.reshape(tf.image.resize(tf.reshape(encoded[i], shape=pixelcnn_input_shape), image_shape[0:2], antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), shape=image_shape[0:2]))
    plt.show()

# visualize_autoencoder(6)

pixelcnn = None
improve_existing = False

if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    if improve_existing:
        pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
    else:
        pixelcnn = get_pixel_cnn(kernel_size=max(pixelcnn_input_shape[0], pixelcnn_input_shape[1]), input_shape=pixelcnn_input_shape[0:2])

    encoded = tf.reshape(model.encoder.predict(
        data["train"][0:len(data["train"])]), shape=(-1, latent_dims))
    indices = get_indices(
        model.vq.variables[0], encoded, quantize=False, splits=8)
    indices = tf.reshape(indices, shape=(-1, *pixelcnn_input_shape[0:2]))
    zeros = tf.zeros_like(indices)

    if not improve_existing:
        pixelcnn.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    pixelcnn.fit(
        x=tf.concat([indices], axis=0),
        y=tf.cast(tf.one_hot(tf.cast(tf.concat([indices], axis=0), dtype=tf.int64),
                  num_embeddings), dtype=tf.float64),
        batch_size=128,
        epochs=30,
        validation_split=0.1)
else:
    pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")

if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    pixelcnn.predict(tf.random.uniform(shape=(1, *pixelcnn_input_shape[0:2]),
                     dtype=tf.int64, maxval=num_embeddings))
    #pixelcnn.save("pixelcnn.ckpt")

print(pixelcnn.summary())

n = 4
generated = data["train"][:n]
generated = tf.zeros_like(generated)
generated = tf.random.uniform(shape=tf.shape(generated), maxval=31)
generated = tf.reshape(generated, shape=(n, *image_shape))
generated = model.encoder.predict(generated)
generated = tf.reshape(generated, shape=(-1, latent_dims))
generated = get_indices(model.vq.variables[0], generated, quantize=False)
generated = tf.reshape(generated, shape=(-1, *pixelcnn_input_shape[0:2]))
generated = generated.numpy()
for _ in range(10):
    for row in range(pixelcnn_input_shape[0]):
        print("Row: ", row)
        for col in range(pixelcnn_input_shape[1]):
            probabilities = pixelcnn.predict(generated)[:, row, col]
            # probabilities = tf.square(probabilities)
            # probabilities = tf.square(probabilities)
            print(probabilities)
            probabilities = tf.math.log(probabilities)

            generated[:, row, col] = tf.reshape(
                tf.random.categorical(probabilities, 1), shape=(n,))

plt.tight_layout()
fig, axs = plt.subplots(n, 2, figsize=image_shape[0:2])
for i in range(n):
    axs[i, 0].imshow(generated[i])
    pred = model.decoder.predict(tf.reshape(tf.gather(model.vq.variables[0], tf.cast(
        generated[i], dtype=tf.int64)), shape=(1, *encoded_image_shape)))
    axs[i, 1].imshow(tf.reshape(pred, shape=image_shape[0:2]))
plt.show()
