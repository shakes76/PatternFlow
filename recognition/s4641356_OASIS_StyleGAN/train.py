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
    style_GAN.track_mean(data_loader.get_mean())


    #Manual Labels to assign to batches when conducting supervised training on discriminator
    # real_labels = tf.ones(shape = (batch_size,))
    # fake_labels = tf.zeros(shape = (batch_size,)) Classification more like not classification ha gottem

    #start with zeros as the background of OASIS images are black
    batch_generator_base = np.repeat(style_GAN._generator_base, batch_size, axis = 0)
    
    #Conduct training
    for e in range(epochs):
        print(">> epoch {}/{}".format(e+1,epochs))
        epoch_metrics = {metric: [] for metric in modules.StyleGAN.METRICS}
        for b in range(num_batches):
            print(">>>> batch {}/{}: ".format(b+1,num_batches), end ="")

            #train discriminator on real images
            # style_GAN._discriminator.trainable = True
            with tf.GradientTape() as tape:
                batch = data_loader.get_data(batch_size)
                real_logits = style_GAN._discriminator(batch)
                discrim_real_loss = tf.math.softplus(-real_logits) #softmax is a curve that restricts to positive. The discriminator returns a single value indicating how "real" it thinks it is. So we try to maximise the distance between real and fake.
                #The more negative a value of real_logits, the closer softmax is to 0 (similar to exponential), this means close to 0 loss.
            style_GAN._discriminator_optimiser.apply_gradients(zip(tape.gradient(discrim_real_loss, style_GAN._discriminator.trainable_variables), style_GAN._discriminator.trainable_variables))  

            #train discriminator on fake images
            with tf.GradientTape() as tape:
                fakes = style_GAN._generator(GANutils.random_generator_inputs(batch_size,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res) + [batch_generator_base])
                fake_logits = style_GAN._discriminator(fakes)
                discrim_fake_loss = tf.math.softplus(fake_logits) #Should be returning a very low "realness score" so our loss is directly the softmax sum of output
            style_GAN._discriminator_optimiser.apply_gradients(zip(tape.gradient(discrim_fake_loss, style_GAN._discriminator.trainable_variables), style_GAN._discriminator.trainable_variables))
            # style_GAN._discriminator.trainable = False

            #Train generator. 
            with tf.GradientTape() as tape:
                gen_logits = style_GAN._gan(GANutils.random_generator_inputs(batch_size,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res) + [batch_generator_base])
                gen_loss =  tf.math.softplus(-gen_logits) #Indirect training of discriminator weights are disabled, so we train the generator weights to make something that outputs 'real' from discriminator #TODO does this flag actually matter at all anymore?
             
            style_GAN._generator_optimiser.apply_gradients(zip(tape.gradient(gen_loss, style_GAN._generator.trainable_variables), style_GAN._generator.trainable_variables))

            #compute the average loss per batch
            metrics_to_store = list(map(lambda x: tf.math.reduce_mean(x).numpy(),[discrim_real_loss, discrim_fake_loss, gen_loss]))
            for m in range(len(modules.StyleGAN.METRICS)):
                epoch_metrics[modules.StyleGAN.METRICS[m]].append(metrics_to_store[m])
            print("{0}: {3}, {1}: {4}, {2}: {5}".format(*(modules.StyleGAN.METRICS + metrics_to_store)))

        style_GAN._epochs_trained += 1
        GANutils.save_training_history(epoch_metrics,training_history_location)
        
        #Save image samples if appropriate
        if image_sample_output is not None:
            samples = style_GAN(GANutils.random_generator_inputs(image_sample_count,style_GAN._latent_dim,style_GAN._start_res,style_GAN._output_res)) #makes use of __call__ defined in styleGAN
            sample_directory = image_sample_output + "epoch_{}".format(style_GAN._epochs_trained)
            GANutils.make_fresh_folder(sample_directory)
            for s,sample in enumerate(samples):
                GANutils.create_image(GANutils.denormalise(sample, style_GAN.get_mean()), sample_directory+"/{}".format(s+1))

        #Save model if appropriate
        if model_output is not None:
            GANutils.make_fresh_folder(model_output)
            style_GAN.save_model(model_output)


def train(model: str = "model/", image_source: str = "images/", epochs: int = 10) -> None:
    #TODO Docstring

    #Constant training parameters (Configured for training locally, using RTX2070)
    BATCH_SIZE = 64
    TRAINING_HISTORY_FILE = "history.csv"
    IMAGE_SAMPLE_COUNT = 5
    IMAGE_SAMPLE_FOLDER = "output/"
    COMPRESSION_SIZE = 32
    
    IMAGE_SIZE = 256
    GENERATOR_INIT_SIZE = 4
    LATENT_DIM = 512

    data_loader = dataset.OASIS_loader(image_source, COMPRESSION_SIZE)
    style_GAN = modules.StyleGAN(COMPRESSION_SIZE,GENERATOR_INIT_SIZE,LATENT_DIM, model) #constants provided to generate a new StyleGAN with the same parameters as the pretrained model if model = None

    #If the user has requested a new model, store it at the default out (otherwise we update the model in specified folder)
    if model == None:
        model = "model/"

    fit(style_GAN, data_loader, batch_size = BATCH_SIZE, epochs = epochs, training_history_location = TRAINING_HISTORY_FILE, image_sample_count = IMAGE_SAMPLE_COUNT, image_sample_output = IMAGE_SAMPLE_FOLDER, model_output = model)

    #Will show plot and also save to disk. Note that by design the history is appended to, so previous runs of the same model (that is the complete training history) will be plotted
    #GANutils.plot_training(GANutils.load_training_history(TRAINING_HISTORY_FILE) ,"training_loss.png", style_GAN._epochs_trained) TODO fix plotting

