#  Problem 6

This library utilizes DCGAN structure in Tensorflow to attempt to produce OASIS brain segment slice images. "The Open Access Series of Imaging Studies (OASIS) is a project aimed at making neuroimaging data sets of the brain freely available to the scientific community" [OASIS](https://www.oasis-brains.org/#about). With this data set availability, the DCGAN model is able to utilise a large dataset of images to constantly perfect it's output image. 

## Dependencies
Before usage instructions can begin, dependencies need to be presented as for the user to install. With your choice of environment initialisers, the following dependencies need to be added:
* Tensorflow / Tensorflow-gpu - If the system you are utilising has a GPU, then Tensorflow-GPU is the best option, otherwise Tensorflow.
* Pillow - Image processing library
* Numpy

## Algorithm Description
As was previously eluded to, this model utilises a DCGAN model. This model is typically known for being fully connected and making use of upsampling and downsampling layers. The following image will detail the general structure for both generator and discriminator models:
![g](https://gluon.mxnet.io/_images/dcgan.png)
[Image](https://gluon.mxnet.io/_images/dcgan.png)

Input images are normalised between [0,1], and the discriminator model constantly downsamples both input images and generated images to a single output layer. This output layers is either 0 or 1, with a 0 indicating a fake image, and a 1 indicating a real image. 

The generator's input is a noise vector. This noise vector is upsampled using leaky relu activation functions until the eventual output shape is achieved, at which point sigmoid is used, to map values for output between [0,1]. 

## Usage
Before usage steps can be listed, it is important to understand how folder structures need to be arranged in order for correct processing. Data that is going to be loaded into the GAN will need to be of the following folder structure:
```
└───keras_png_slices_data
    └───keras_png_slices_data
        ├───keras_png_slices_seg_test
        ├───keras_png_slices_seg_train
        ├───keras_png_slices_seg_validate
        ├───keras_png_slices_test
        ├───keras_png_slices_train
        └───keras_png_slices_validate
```
Furthermore, the parent directory of "keras_png_slices_data" must not contain any files/folders with the name of "gen_images". If there does exist such directory, all contents will be erased when this model is run. With this established, usage steps can now begin. Take for example, the above folder structure is located in :
```
C:\Users\lains\Downloads
```
Now, navigating to the directory of which this model is located, the following command must be run:
```
python main.py "C:\Users\lains\Downloads"
```

## Results
The following results were achieved for their respective epoch:
| Epoch| Produced Image
:-------------------------:|:-------------------------:
1 epoch | ![y](https://i.ibb.co/S6HvnDV/generated-img1.png) 
235 epoch | ![x](https://i.ibb.co/8N8Z9qT/generated-img235.png)  
515 epoch | ![g](https://i.ibb.co/c8KdSFz/generated-img515.png) 
550 epoch | ![y](https://i.ibb.co/MR3Dr12/generated-img550.png)
551 epoch | ![s](https://i.ibb.co/WWYg55Y/generated-img551.png) 


## References
[Tensorflow](https://www.tensorflow.org/tutorials/generative/dcgan)

[Jeff Heaton](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb)