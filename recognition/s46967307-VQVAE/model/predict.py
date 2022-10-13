import os
from tkinter import W
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import matplotlib.pyplot as plt
from modules import AE, get_pixel_cnn, latent_dims, get_indices
from dataset import load_data

model: AE = tf.keras.models.load_model("model.ckpt")
print(model.summary())

data = load_data()

#predictions = model.predict(data["test"])
#encoded = get_indices(model.vq.variables[0], tf.reshape(model.encoder.predict(data["test"]), shape=(-1,latent_dims)))
#encoded = tf.reshape(encoded, shape=(-1, 32, 32))
#
#n = 30
#plt.tight_layout()
#fig, axs = plt.subplots(n, 2, figsize=(256,256))
#for i in range(n):
#    axs[i,0].imshow(predictions[i])
#    axs[i,1].imshow(encoded[i])
#plt.savefig("out.png", dpi=50)
#exit()

pixelcnn = None

if not os.path.exists("pixelcnn.ckpt"):
    pixelcnn = get_pixel_cnn(kernel_size=32, input_shape=(32,32,1))

    # We have to quarter the amount of training data to avoid running out of vram (I think)
    encoded = tf.reshape(model.encoder.predict(data["train"][0:int(len(data["train"])/4)]), shape=(-1, latent_dims))
    indices = get_indices(model.vq.variables[0], encoded)
    indices = tf.reshape(indices, shape=(-1,32,32))

    pixelcnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    pixelcnn.fit(
        x=indices,
        y=indices,
        batch_size=8,
        epochs=3,
        validation_split=0.1
    )
else:
    pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
    
print(pixelcnn.summary())

# Testing PixelCNN for random noise to brain:
n = 5
plt.tight_layout()
fig, axs = plt.subplots(n, 3, figsize=(256,256))
for i in range(n):
    encoded = get_indices(model.vq.variables[0], 
        tf.reshape(model.encoder.predict(data["train"][0:int(len(data["train"])/4)]), shape=(-1,latent_dims)))
    indices = tf.reshape(encoded, shape=(-1, 32, 32, 1))
    #indicies = tf.zeros_like(indices)

    #indices = tf.random.uniform(shape=(1,32,32,1), dtype=tf.int64, maxval=32)
    predicted_embeddings = pixelcnn.predict(tf.expand_dims(indices[n], 0))
    predicted_embeddings = tf.math.round(predicted_embeddings)
    predicted_embeddings = tf.cast(predicted_embeddings, dtype=tf.int64)
    embeddings = tf.gather(model.vq.variables[0], predicted_embeddings)
    embeddings = tf.reshape(embeddings, shape=(1,32,32,8))
    prediction = model.decoder.predict(embeddings)
    axs[i,0].imshow(tf.reshape(indices[n], shape=(32,32)))
    axs[i,1].imshow(tf.reshape(predicted_embeddings, shape=(32,32)))
    axs[i,2].imshow(tf.reshape(prediction, shape=(256,256)))

plt.savefig("out.png", dpi=50)

if not os.path.exists("pixelcnn.ckpt"):
    pixelcnn.save("pixelcnn.ckpt")