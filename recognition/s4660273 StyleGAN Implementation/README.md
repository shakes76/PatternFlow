
# Style GAN2 implementation for OASIS Brain dataset
### s4660273 Sahil Tumbare

![StyleGAN Output](https://github.com/sahiltumbare/PatternFlow/blob/5a42f1ad110a320ed436ff3b490dd8e21eda05fc/recognition/s4660273%20StyleGAN%20Implementation/StyleGAN2%20sample.png?raw=true)

**Requirements**

Requirements are the same as for StyleGAN2:

* 64-bit Python 3.6 installation. Recommend to use Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.14 (Windows and Linux) TensorFlow 2.x is not supported.
* One or more high-end NVIDIA GPUs with at least 12GB DRAM, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5.

**Introduction**

StyleGAN2 is the improved version of a GAN. This neural network model is found effective with great precision to regenerate the images with respect to the trained daaset. Training it with wide variety of data can help to regenrate the corrupted or damaged images. In case of MRI the images are complex. So, regeneration is equally laborous. In such cases the StyleGAN2 can help to achieve this with the great accuracy.

StyleGAN conains 2 main components:
**1.  Generator**

Generator is responsible for creation of the fake images from the random data space. Its task is to create the images which can fool the Discriminator so that the fake image is classified as a real one.

**2.  Discriminator**

 Whereas, the discriminator is responsible for identifying the fake images from the input images. It learns the special features from the training dataset and based on it classifies the unseen input images.

![StyeGAN2 Architecture](https://github.com/sahiltumbare/PatternFlow/blob/5a42f1ad110a320ed436ff3b490dd8e21eda05fc/recognition/s4660273%20StyleGAN%20Implementation/architecture.png?raw=true)Fig. Generator Architecture

**The incremental list of changes to the generator are:**

1. Baseline Progressive GAN.

Progressive growing GAN training method is used for training of Generator and Discriminator. It means that the it starts with small images, like 4x4, and progresively increases the width and height of an image in the power of two. Post which, adaptive instance normalization (aka AdaIN) is used to transform and encorporate the style vector in every block of the generator.

1. Tuning and bilinear upsamplings are added.

Instead of transpose convolutional layers, used in other generator networks,
nearest neighbour layers are used for upsampling in progressive GAN. Whereas, in styleGAN bilinear upsampling layers are used. 

1. Mapping network and AdaIN (styles) have been done.

Randomlysampled points are taken from the latent space and style vector is created with the help of mapping network(comprised of eight fully connected layers).

![diagram](https://github.com/sahiltumbare/PatternFlow/blob/cfca87fc814fe189892de7ac4ff3eb7ce6f23fb7/recognition/s4660273%20StyleGAN%20Implementation/Calculation-of-the-adaptive-instance-normalization-AdaIN-in-the-StyleGAN-300x55.png?raw=true) 
*Fig. Calculation of the adaptive instance normalization (AdaIN) in the StyleGAN.*

1. Removal of latent vector input to generator.

GAN Model is modified in such a way that is does not take input directly from latent space. Instead, it uses constant 4x4x512 value input to start generating images.

1. Added noise in each block.

Gaussian noise is used to incorporate style level variation at each level. 

1. Mixing regularization.

Two style vectors are generated from the mapping network. Split point in the network is chosen then first style vector is used in a AdaIN before split point and all AdaIN operations after split point receives second style vector. It localizes the style to the specific parts in the model along with the level of detais in the generated image.

**Using the code**
**Latent space: Projecting images**
# Project generated images
        python run_projector.py project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl \ --seeds=0,1,5

# Project real images
        python run_projector.py project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl \ --dataset=car --data-dir=~/datasets


**Running of dataset**
Create custom datasets by placing all training images under a single directory. The images must be square-shaped and they must all have the same power-of-two dimensions. To convert the images to multi-resolution TFRecords, run:

	python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images
	python dataset_tool.py display ~/datasets/my-custom-dataset

**Train Model**

	python run_training.py --num-gpus=8 --data-dir=datasets --config=config-e --dataset=my-custom-dataset --mirror-augment=true



