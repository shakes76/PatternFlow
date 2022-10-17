import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from modules import AE, get_pixel_cnn, latent_dims, num_embeddings, get_indices, pixelcnn_input_shape, image_shape, encoded_image_shape, ssim_loss
from dataset import load_data

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)

model: AE = tf.keras.models.load_model("model.ckpt", compile=False)
print(model.summary())

data = load_data()

def visualize_autoencoder(n):
    predictions = model.predict(tf.reshape(data["test"][0:30], shape=(-1,*image_shape[0:2])))
    encoded = get_indices(model.vq.variables[0], tf.reshape(model.encoder.predict(data["test"]), shape=(-1,latent_dims)), quantize=False, splits=32)
    encoded = tf.reshape(encoded, shape=(-1, *pixelcnn_input_shape[0:2]))
    plt.tight_layout()
    fig, axs = plt.subplots(n, 3, figsize=image_shape[0:2])
    for i in range(n):
        axs[i,0].imshow(tf.reshape(data["test"][i], shape=image_shape[0:2]))
        axs[i,2].imshow(predictions[i])
        axs[i,1].imshow(tf.reshape(encoded[i], shape=pixelcnn_input_shape))
    plt.show()

visualize_autoencoder(6)

def calculate_ssim():
    test_data = tf.reshape(data["test"], shape=(-1,*image_shape[0:2]))

    results = model.predict(test_data)

    ssim = tf.image.ssim(tf.expand_dims(test_data, axis=-1), results, max_val=1.0)

    return (tf.reduce_mean(ssim).numpy(), tf.math.reduce_sum(tf.cast(tf.math.greater(ssim,0.6), dtype=tf.float64)).numpy() / len(data["test"]))

mean, pct = calculate_ssim()
print("Average SSIM:", mean, " Percent >0.6:", pct)

pixelcnn = None
improve_existing = False

if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    if improve_existing:
        pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
    else:
        pixelcnn = get_pixel_cnn(kernel_size=max(pixelcnn_input_shape[0], pixelcnn_input_shape[1]), input_shape=pixelcnn_input_shape[0:2])

    if not improve_existing:
        pixelcnn.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    
    for i in range(2):
        dataset = tf.data.Dataset.from_tensor_slices(tf.concat([data["train"], data["validate"]], axis=0)[(i)*3*1024:(i+1)*3*1024])
        dataset = dataset.batch(32)
        pred = None
        for batch in dataset:
            if pred is None:
                pred = model.encoder.predict(batch)
            else:
                pred = tf.concat([pred, model.encoder.predict(batch)], axis=0)
        
        encoded = tf.reshape(pred, shape=(-1, latent_dims))
        indices = get_indices(model.vq.variables[0], encoded, quantize=False, splits=64)
        indices = tf.reshape(indices, shape=(-1, *pixelcnn_input_shape[0:2]))

        pixelcnn.fit(
            x=tf.concat([indices], axis=0),
            y=tf.reshape(tf.cast(tf.one_hot(tf.cast(tf.concat([indices], axis=0), dtype=tf.int64),
                    num_embeddings), dtype=tf.float64), shape=(-1,*pixelcnn_input_shape[0:2],num_embeddings)),
            batch_size=64,
            epochs=60,
            validation_split=0.1)
else:
    pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")

if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    pixelcnn.predict(tf.random.uniform(shape=(1, *pixelcnn_input_shape[0:2]),
                     dtype=tf.int64, maxval=num_embeddings))

print(pixelcnn.summary())

n = 8
generated = tf.zeros(shape=(n,*pixelcnn_input_shape[0:2]))
generated = tf.reshape(generated, shape=(n, *pixelcnn_input_shape[0:2]))
generated = generated.numpy()
for _ in range(1):
    for row in range(pixelcnn_input_shape[0]):
        print("Row: ", row)
        for col in range(pixelcnn_input_shape[1]):
            probabilities = pixelcnn.predict(generated)[:, row, col]
            print(probabilities)

            generated[:, row, col] = tf.reshape(
                tf.random.categorical(probabilities, 1), shape=(n,))

plt.close()
plt.tight_layout()
fig, axs = plt.subplots(n, 2, figsize=image_shape[0:2])
for i in range(n):
    axs[i, 0].imshow(generated[i])
    pred = model.decoder.predict(tf.reshape(tf.gather(model.vq.variables[0], tf.cast(
        generated[i], dtype=tf.int64)), shape=(1, *encoded_image_shape)))
    axs[i, 1].imshow(tf.reshape(pred, shape=image_shape[0:2]))
plt.show()
