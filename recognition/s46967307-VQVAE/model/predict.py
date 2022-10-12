import tensorflow as tf
import matplotlib.pyplot as plt
from modules import AE, GAN
from dataset import load_data

model = tf.keras.models.load_model("model.ckpt")

#n = 5
#plt.tight_layout()
#fig, axs = plt.subplots(n, 2, figsize=(256,256))
#for i in range(n):
#    noise = tf.random.uniform(shape=(1,32,32,1), dtype=tf.int64, maxval=32)
#    noise2 = tf.random.uniform(shape=(1,32,32,1), dtype=tf.int64, maxval=32)
#    noise = tf.reshape(tf.gather(model.vq.get_layer("vq").embeddings, noise), shape=(1,32,32,8))
#    noise2 = tf.reshape(tf.gather(model.vq.get_layer("vq").embeddings, noise2), shape=(1,32,32,8))
#    axs[i,0].imshow(tf.reshape(model.decoder.predict(noise), shape=(256,256)))
#    axs[i,1].imshow(tf.reshape(model.decoder.predict(noise2), shape=(256,256)))
#plt.savefig("out.png", dpi=50)

gan = GAN(model)
gan.compile(optimizer='adam')
print(gan.generator.summary())
print(gan.discriminator.summary())

data = load_data()
gan.fit(data["train"],
        data["train"],
        epochs=2,
        shuffle=True,
        batch_size=8)