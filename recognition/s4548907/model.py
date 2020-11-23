"""
Main function of the DCGAN:
    Generator
    Discriminator
    trian setp (generate images and save, compute ssim)
"""
import tensorflow as tf
from tensorflow.keras import layers
import os

# Build generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (2,2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # in DCGAN the activation function is tanh
    model.add(layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 1)
    
    return model

# This is a part to show the sturcture (layer) of generator
generator = make_generator_model()
generator.summary()

# test if the generator model works
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# Build discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',input_shape=[64, 64, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# test if the discriminator model works
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

#more parameter of the model
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# define the loss function of generator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# define the loss function of discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Set the optimizer parameters
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save the model
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 4

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
test_seed = tf.random.normal([32, noise_dim])

# Define Train and Generate_save images
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)  # the result of discriminator
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input, path_save=None):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        ax = plt.subplot(2, 2, i+1)
        ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        ax.axis('off')
    fig.suptitle("Image in the {:04d}".format(epoch))
    
    if path_save is not None:
        plt.savefig(path_save,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.show()
    else:
        plt.show()

def get_ssim(model):
    
    fake_imgs = model(test_seed, training=False)
    real_imgs =  tf.image.convert_image_dtype(tf_train[0:50], dtype =tf.float32)
    
    all_ssim = tf.image.ssim(fake_imgs, real_imgs, max_val=2)
    mean = tf.math.reduce_max(all_ssim) # take the maximum value of the ssim
    ssim = mean.numpy()
    
    return ssim


def train(dataset, epochs):
    ssim =[]
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)  #train each batch

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
       
        ## save images 50 epoches
        if epoch % 50 == 0:
            path_save = "Result/image_at_epoch_{:04d}.png".format(epoch)
        else:
            path_save=None
            
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,
                                path_save=path_save)

        # Save the model every 15 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        ssim_epoch = get_ssim(generator)
        ssim.append(ssim_epoch)
        
        print ('Time for epoch {} is {} sec, ssim is {}'.format(epoch + 1, time.time()-start, ssim_epoch))
        
        ssim_epoch = get_ssim(generator)
        ssim.append(ssim_epoch)
        

  # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)
    return ssim
    

