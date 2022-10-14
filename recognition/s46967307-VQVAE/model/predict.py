import os
from tkinter import W
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
#import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from modules import AE, get_pixel_cnn, latent_dims, num_embeddings, get_indices
from dataset import load_data

model: AE = tf.keras.models.load_model("model.ckpt")
print(model.summary())

data = load_data()

# Use this code if you want to visualize encodings of actual brains
#predictions = model.predict(tf.reshape(data["test"][0:30], shape=(-1,256,256)))
#encoded = get_indices(model.vq.variables[0], tf.reshape(model.encoder.predict(data["test"]), shape=(-1,latent_dims)), quantize=False)
#encoded = tf.reshape(encoded, shape=(-1, 32, 32))
#
#n = 5
#plt.tight_layout()
#fig, axs = plt.subplots(n, 2, figsize=(256,256))
#for i in range(n):
#    axs[i,0].imshow(predictions[i])
#    axs[i,1].imshow(tf.reshape(tf.image.resize(tf.reshape(encoded[i], shape=(32,32,1)), (256,256), antialias=False), shape=(256,256)))
#plt.savefig("out.png", dpi=50)
#exit()

pixelcnn = None
improve_existing = False

if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    if improve_existing:
        pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
    else:
        pixelcnn = get_pixel_cnn(kernel_size=32, input_shape=(32,32))

    encoded = tf.reshape(model.encoder.predict(data["train"]), shape=(-1, latent_dims))
    indices = get_indices(model.vq.variables[0], encoded, quantize=False, splits=8)
    indices = tf.reshape(indices, shape=(-1,32,32))

    if not improve_existing:
        pixelcnn.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics = ["accuracy"])
    pixelcnn.fit(
        x=indices,
        y=tf.cast(tf.one_hot(tf.cast(indices, dtype=tf.int64), num_embeddings), dtype=tf.float64),
        batch_size=128,
        epochs=20,
        validation_split=0.1)
else:
    pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
    
if not os.path.exists("pixelcnn.ckpt") or improve_existing:
    pixelcnn.predict(tf.random.uniform(shape=(1,32,32), dtype=tf.int64, maxval=num_embeddings))
    pixelcnn.save("pixelcnn.ckpt")

print(pixelcnn.summary())

inpt = data["train"][0]
# inpt = tf.zeros_like(inpt)
# inpt = tf.ones_like(inpt)
inpt = tf.reshape(inpt, shape=(1,256,256,1))
inpt = model.encoder.predict(inpt)
inpt = tf.reshape(inpt, shape=(-1,latent_dims))
inpt = get_indices(model.vq.variables[0], inpt, quantize=False)
inpt = tf.reshape(inpt, shape=(-1,32,32))
out = pixelcnn.predict(inpt)
n = 6
plt.tight_layout()
fig, axs = plt.subplots(n, n, figsize=(256,256))
for i in range(n):
    for j in range(n):
        if i*n+j < 32:
            axs[i,j].imshow(out[0,:,:,i*n + j])
plt.show()

generated = tf.zeros(shape=(4,32,32)).numpy()
generated = data["train"][:4]
generated = tf.reshape(generated, shape=(4,256,256,1))
generated = model.encoder.predict(generated)
generated = tf.reshape(generated, shape=(-1,latent_dims))
generated = get_indices(model.vq.variables[0], generated, quantize=False)
generated = tf.reshape(generated, shape=(-1,32,32))
generated = generated.numpy()
for row in range(32):
    print(row, "/ 32")
    for col in range(32):
        probabilities = pixelcnn.predict(generated)[:, row, col]
        #probabilities = tf.square(probabilities)
        #print(probabilities)
        probabilities = tf.math.maximum(probabilities, tf.constant([0.0]))
        probabilities = tf.math.log(probabilities)

        #out = pixelcnn.predict(generated)
        #n = 6
        #plt.tight_layout()
        #fig, axs = plt.subplots(n, n, figsize=(256,256))
        #for i in range(n):
        #    for j in range(n):
        #        if i*n+j < 32:
        #            axs[i,j].imshow(out[0,:,:,i*n + j])
        #axs[5,5].imshow(generated[0])
        #plt.imshow(generated[0])
        #plt.show()

        generated[:, row, col] = tf.reshape(tf.random.categorical(probabilities, 1), shape=(4,))

n = 4
plt.tight_layout()
fig, axs = plt.subplots(n, 2, figsize=(256,256))
for i in range(n):
    axs[i,0].imshow(generated[i])
    pred = model.decoder.predict(tf.reshape(tf.gather(model.vq.variables[0], tf.cast(generated[i], dtype=tf.int64)), shape=(1,32,32,latent_dims)))
    axs[i,1].imshow(tf.reshape(pred, shape=(256,256)))
plt.show()

## Testing PixelCNN for random noise to brain:
#n = 5
#plt.tight_layout()
#fig, axs = plt.subplots(n, 3, figsize=(256,256))
#for i in range(n):
#    encoded = get_indices(model.vq.variables[0], 
#        tf.reshape(model.encoder.predict(data["train"][0:int(len(data["train"])/4)]), shape=(-1,latent_dims)))
#    indices = tf.reshape(encoded, shape=(-1, 32, 32, 1))
#    #indicies = tf.zeros_like(indices)
#
#    #indices = tf.random.uniform(shape=(1,32,32,1), dtype=tf.int64, maxval=32)
#    predicted_embeddings = pixelcnn.predict(tf.expand_dims(indices[n], 0))
#    predicted_embeddings = tf.math.round(predicted_embeddings)
#    predicted_embeddings = tf.cast(predicted_embeddings, dtype=tf.int64)
#    embeddings = tf.gather(model.vq.variables[0], predicted_embeddings)
#    embeddings = tf.reshape(embeddings, shape=(1,32,32,8))
#    prediction = model.decoder.predict(embeddings)
#    axs[i,0].imshow(tf.reshape(indices[n], shape=(32,32)))
#    axs[i,1].imshow(tf.reshape(predicted_embeddings, shape=(32,32)))
#    axs[i,2].imshow(tf.reshape(prediction, shape=(256,256)))
#
#plt.savefig("out.png", dpi=50)
