import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from vqvae import VQVAE 

if __name__ == "__main__":
    IMG_SIZE = 256

    # load OASIS images from folder
    dataset = image_dataset_from_directory("keras_png_slices_data/keras_png_slices_train", 
                                          label_mode=None, 
                                          image_size=(IMG_SIZE, IMG_SIZE),
                                          color_mode="grayscale",
                                          batch_size=32)

    # normalize pixels between [0,1]
    dataset = dataset.map(lambda x: (x / 255.0))

    # calculate variance of training data (at a individual pixel level) to pass into VQVAE
    count = dataset.unbatch().reduce(tf.cast(0, tf.int64), lambda x,_: x + 1 ).numpy()
    mean = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + y ).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE)
    var = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + tf.math.pow(y - mean,2)).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE - 1)

    # initial hyperparameters to get it working. 
    # Need to be tuned!
    learning_rate = 2e-4
    beta = 0.25
    latent_dim = 64
    num_embeddings = 128
    epochs = 10
    batch_size = 128

    input_size = (IMG_SIZE, IMG_SIZE, 1)

    # create model
    vqvae_model = VQVAE(input_size, latent_dim, num_embeddings, beta, var)
    vqvae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # fit it
    vqvae_model.fit(dataset, epochs=epochs, batch_size=batch_size)
