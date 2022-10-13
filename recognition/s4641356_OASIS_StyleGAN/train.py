from matplotlib import style
import tensorflow as tf
import numpy as np
import GANutils
import modules
import dataset


def fit(style_GAN: modules.StyleGAN, data_loader: dataset.OASIS_loader, batch_size: int, epochs: int, training_history_location: str, image_sample_count:int = 5, image_sample_output:str = "output/", model_output:str = "model/") -> None:
    """
        Fits the model to a given set of training images. The custom GAN training paradigm is as follows: TODO
    Args:
        style_GAN (modules.StyleGAN): styleGAN to fit on given images
        data_loader (dataset.OASIS_loader): dataloader to draw image data from
        batch_size (int): number of images to pass through model per batch
        epochs (int): number of epochs (repeat runs of full training set) to train model on
        training_history_location (str): filepath of csv (no file extension) to save training history
        image_sample_count (int, optional): Number of image samples to take per epoch ignored if image_sample_output is set to None. Defaults to 5
        image_sample_output (str, optional): Directory to save sample images after each epoch, or set to None to disable saving. Defaults to "output/". TODO Go back to all folder args and rework to not require trailing /
        model_output (str, optional): directory save model information after each epoch, or set to None to disable model saving. Defaults to "model/". 
    """
    #TODO clean up the mass amount of encapsulation violation
    #TODO sanitse inputs
    num_batches = data_loader.get_num_images_availible()//batch_size #required batches per epoch (we use just under the full set if there is a remainder, this prevents duplicates per epoch)
    
    #Manual Labels to assign to batches when conducting supervised training on discriminator
    real_labels = tf.ones(shape = (batch_size,))
    fake_labels = tf.zeros(shape = (batch_size,))

    #start with zeros as the background of OASIS images are black
    batch_generator_base = np.repeat(style_GAN._generator_base, batch_size, axis = 0)
    
    #Conduct training
    for e in range(epochs):
        print(">> epoch {}/{}".format(e+1,epochs))
        epoch_metrics = {metric: [] for metric in modules.StyleGAN.METRICS}
        for b in range(num_batches):
            print(">>>> batch {}/{}: ".format(b+1,num_batches), end ="")

            batch = data_loader.get_data(batch_size)

            #train discriminator on real images
            dfl, dfa = style_GAN._discriminator.train_on_batch(batch, real_labels)

            #train discriminator on a batch of fake images from current iteration of the generator
            fake_images = style_GAN._generator(GANutils.random_generator_inputs(batch_size,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res) + [batch_generator_base])
            drl, dra = style_GAN._discriminator.train_on_batch(fake_images, fake_labels)

            #Train generator. Indirect training of discriminator weights are disabled, so we train the generator weights to make something that outputs 'real' (1) from discriminator
            gl, ga = style_GAN._gan.train_on_batch(GANutils.random_generator_inputs(batch_size,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res) + [batch_generator_base], real_labels)

            metrics_to_store = [dfl,dfa,drl,dra,gl,ga]
            for m in range(len(modules.StyleGAN.METRICS)):
                epoch_metrics[modules.StyleGAN.METRICS[m]].append(metrics_to_store[m])

            print(metrics_to_store)

        style_GAN._epochs_trained += 1
        GANutils.save_training_history(epoch_metrics,training_history_location)
        
        #Save image samples if appropriate
        if image_sample_output is not None:
            samples = style_GAN(GANutils.random_generator_inputs(image_sample_count,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res)) #makes use of __call__ defined in styleGAN
            sample_directory = image_sample_output + "epoch_{}".format(style_GAN._epochs_trained)
            GANutils.make_fresh_folder(sample_directory)
            for s,sample in enumerate(samples):
                GANutils.create_image(GANutils.denormalise(sample,data_loader.get_mean()), sample_directory+"/{}".format(s+1))

        #Save model if appropriate
        if model_output is not None:
            GANutils.make_fresh_folder(model_output)
            style_GAN.save_model(model_output)