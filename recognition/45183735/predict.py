import tensorflow as tf
import train
import modules
import dataset
import matplotlib.pyplot as plt


class Predict:
    def __init__(self, train_model):
        self.train_model = train_model

    def predict(self):
        # generate a seed
        seed = [tf.random.uniform([self.train_model.sample_num, self.train_model.g_model.latent_size]) for i in
                range(7)] \
               + [tf.random.uniform([self.train_model.sample_num, 4 * 2 ** i, 4 * 2 ** i, 1]) for i in range(7)]
        # make prediction using the model
        predictions = self.train_model.g_model.model(seed, training=False)

        # plot the prediction
        plt.figure(figsize=(4, 4), constrained_layout=True)
        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow((predictions[i].numpy()), cmap="gray")
            plt.axis("off")
        plt.savefig("predict.png", dpi=500)
        plt.show()
        plt.close()

    def load_latest_trained_model(self, ckpt_dir):
        self.train_model.ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()


if __name__ == "__main__":
    latent_size = 512
    input_size = 256
    batch_size = 8
    # mapping network for generator
    g_mapping = modules.G_Mapping(latent_size)
    # synthesis network for generator
    g_s = modules.G_Synthesis(latent_size, g_mapping, input_size)
    # generator model
    g_style = modules.G_style(latent_size, input_size, g_s)
    # discriminator model
    discriminator = modules.Discriminator(input_size)
    # dataset
    dataset = dataset.Dataset("./keras_png_slices_data", batch_size, input_size)
    # train model
    t = train.Train(dataset.train_ds, g_style, discriminator, input_size, batch_size)
    # prediction model
    prediction = Predict(t)
    ckpt_dir = "./checkpoint"
    # load the trained model
    prediction.load_latest_trained_model(ckpt_dir)
    # make prediction
    prediction.predict()
