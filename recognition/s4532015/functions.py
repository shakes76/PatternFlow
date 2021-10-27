#functions required by other files
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from ModConv2D import ModConv2D
from tqdm import tqdm
import imageio
import glob


def noise(n, latent_size):
    return tf.random.normal([n, latent_size])


def noiseList(n, n_layers, latent_size):
    return [noise(n, latent_size)] * n_layers

def crop_to_fit(x):
    #makes sure that x[0] has the same dimensions as x[1]
    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]

def make_output_size(s1, s2):
    ss = int(s2 / s1)
    def upsample_to_size(x, y = ss):
        x = K.resize_images(x, y, y, "channels_last", interpolation='bilinear')
        return x
    return upsample_to_size

def to_output(inputs, style, im_size):
    #want to do a ModConv2D on input with styles like normal
    size = inputs.shape[2]
    x = ModConv2D(1, 1, kernel_initializer=VarianceScaling(200/size), demod=False)([inputs, style])
    #upsample image to be (None, im_size, im_size, None)
    return Lambda(make_output_size(size, im_size), output_shape=[None, im_size, im_size, None])(x)

#make gif of the generated outputs from each epoch
def make_gif():
    anim_file = 'epoch_ouputs.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


#train functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr, axis=tf.range(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight

#generate and save images
def generate_and_save_images(model, epoch, test_input, batch_size, save=True):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(12, 12))

    for i in range(batch_size):
        plt.subplot(2, batch_size//2, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.subplots_adjust(hspace=-0.6, wspace=0.1)
    if save: plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()


#define training step
@tf.function
def train_step(S, G, D, gen_model, batch_size, im_size, gen_model_optimiser, D_optimiser, images, style, noise, pl_mean):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #get style vectors
        w = []
        for i in range(len(style)):
            w.append(S(style[i]))
        #generated images
        gen_images = G((w, noise))
        
        #discriminate between the two
        more_noise = tf.random.normal([batch_size, im_size, im_size, 1])*0.01
        real_output = D(images + more_noise, training = True)
        fake_output = D(gen_images, training=True)

        #loss functions
        gen_loss = K.mean(fake_output)
        disc_loss = K.mean(K.relu(1+real_output) + K.relu(1 - fake_output))*0.9
        
        #gradient penalty
        disc_loss += gradient_penalty(images, real_output, 10)

        #path length regularisation
        w_2 = []
        for i in range(len(style)):
            #slightly adjust w
            std = 0.1 / (K.std(w[i], axis = 0, keepdims = True) + 1e-8)
            w_2.append(w[i] + K.random_normal(tf.shape(w[i])) / (std + 1e-8))
        #generate second set of images
        pl_images = G((w_2, noise))

        #get path length
        delta_g = K.mean(K.square(pl_images - gen_images), axis = [1, 2, 3])
        pl_lengths = delta_g

        if pl_mean > 0:
            gen_loss += K.mean(K.square(pl_lengths - pl_mean))

    #get gradients
    gen_gradients = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

    #apply gradients
    gen_model_optimiser.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
    D_optimiser.apply_gradients(zip(disc_gradients, D.trainable_variables))

    return disc_loss, gen_loss, pl_lengths


#define training loop
def train(S, G, D, gen_model, gen_model_optimiser, D_optimiser, batch_size, im_size, n_layers, latent_size, dataset, epochs):
    disc_loss = []
    gen_loss = []
    pl_mean = 0
    #make the seed for output images
    z = noiseList(batch_size, n_layers, latent_size)
    added_noise = tf.random.normal((batch_size, im_size, im_size, 1))
    seed = (z, added_noise)

    for epoch in range(epochs):
        temp_disc_loss = 0
        temp_gen_loss = 0
        for image_batch in tqdm(dataset, desc=f"epoch {epoch + 1}"):
            #train step
            style = noiseList(batch_size, n_layers, latent_size) #z
            noise = tf.random.normal([batch_size, im_size, im_size, 1]) #B
            a, b, d = train_step(S, G, D, gen_model, batch_size, im_size, gen_model_optimiser, D_optimiser, image_batch, style, noise, pl_mean)
            
            #adjust path length mean
            if pl_mean == 0:
                pl_mean = K.mean(d)
            pl_mean = 0.9*pl_mean + 0.1*K.mean(d)

            #save the train step losses
            temp_disc_loss += a
            temp_gen_loss += b
        #average the losses for each epoch
        disc_loss.append(temp_disc_loss/(9664/batch_size))
        gen_loss.append(temp_gen_loss/(9664/batch_size))

        #generate a plot of the outcomes from this epoch and save it
        generate_and_save_images(gen_model, epoch + 1, seed, batch_size)

        #print info
        print("gen loss:", gen_loss[-1].numpy())
        print("disc loss:", disc_loss[-1].numpy())
    return disc_loss, gen_loss