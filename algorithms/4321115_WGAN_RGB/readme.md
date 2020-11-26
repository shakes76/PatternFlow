# keras implementation of the WGAN for colored images

## Introduction 

WGAN networks were argued to have higher quality output thanks to the use of the Wasser-Stein loss, which:
. avoids the vanishing/exploding gradient issue normally associated with the KL/JA divergence losses
. Adds a score to the realism of an image, as opposed to the sigmoid binary output which effectively performs classification. 

### Critic

The critic is a traditional classifier using a conv2D > batchNormalisation > LeakyReLU activation starting with an input of mxnc tensor over a specified number of blocks before flattening to a dense layer and outputting using a sigmoid activation.

### Generator

The generator takes a laten space vector, uniform random values of domain [0, 1], shaped into a tensor of 4x4xfeatures that passes through blocks of upsampling, convolutions, batchNormalisation, and LeakyReLU activations before the output layer which gives a generated image of the required size, starting at 64 features, and increasing until the required size, augmenting the feature space.

### WGAN

The WGAN stacks the generator and the critic. The critic's training is turned off, as it is done with a 5:1 ratio per step, dividing the batches per epoch for both critic and generator. The critic effectively trains 5 times more than the generator, to give it a more robust discriminant function, while the generator trains in the iteration, and is trained equally through the WGAN.

Keep in mind that the WGAN model has a 5:1 ratio between critic training and generator training. This parameter can be tuned, but papers recommend this ratio.

The paper is available here(https://arxiv.org/abs/1701.07875)

The pros for considering this model are:
. 150 epochs yield some reasonably good results
. The network is scalable

Some of the cons from experiencing this model are:
. loss gives no pertinent information about the quality of the generated image, so considering the distance of 2 distributions has no relevance and may require an extra function such as using critical t value from the differentiation of distributions. 
. The model is very sensitive to hyper-parameters
. In terms of real applications, this is not as useful as an image to image generation model such as the cycle GAN or the style GAN.

## Dependencies

This code depends on the Celeba-dataset available here(http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Once downloaded, place in the root directory under "./celeba-dataset"

note: you will need to login first, and download the set using an auth key. Also, this data set is large (over 1.5 gb), so make sure to start with smaller numbers of samples if you are on single CPU setups or have restricted memory space access on your server.

note II: This setup was made particularly for the celeba dataset, however replacing the default path argument in load_images() will take any image directory for pre-processing.

The python dependencies are:
1. tensorflow and keras
2. numpy and scipy
3. opencv-python
4. pydub and pyttsx3 (if you intend to use the sonify.py tools)

You can run the ./dependencies.sh script to pip install the required modules

# Hyperparameters

Currently, the defaults are:
n_samples = 3000 randomised images from the celeba data-set
n_epochs = 150 (usually some good definition can be seen at this point)
the random uniform initialisation of the kernel for both critic and generator are set to a standard deviation of 0.2
batchNormalisation has a value of 0.2
The learning rate is set to 0.00005
The laten space vector is set to 100

#Output

The completed training will result in several outputs:
. the images directory with generated intermittent color images
. the metrics.csv output 
. A plot of the loss for critic and WGAN

##Optional

Once the training is completed, you can run sonify.sh to run a sonification of the generated images to run some overall sound based debugging of the images. Not sure if this is useful outside the scope of blind users, but something that is available.

#Discussion

the expansion of the generator uses upsampling and convolution layers back to back in order to increase the feature space of the image, and allow the generator to be more "creative". perhaps some expansions on individual channels of this could give some useful results. This can be seen on the EfficientNet architecture here(https://arxiv.org/abs/1905.11946), where an expansion/squeeze phase runs separate feature convolution on the RGB channels separately before merging them back before the fully connected layer.

There are discussions about the initial latent space, which could be augmented by giving an initial color direction, feature galvanisation and placement of the for the model to be coaxed by user preferences.

Also, the addition of an alpha channel might make such a WGAN more applicable to innovation, such as allowing dynamic insertions of various objects in 2D web spaces or 3D spaces, or mixed media to include various quickly generated sprites/labels/diagram objects that are relative to a given topic.
