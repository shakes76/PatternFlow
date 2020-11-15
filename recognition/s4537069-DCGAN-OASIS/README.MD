# Generative DCGAN Model for OASIS Brain Dataset using TF
This model generates brain scans using the OASIS Brain dataset through the use of a Direct Convolutional Generative Adversial Network (DCGAN). The model is fairly successful, evident using a Structure Similarity (SSIM) comparison. The SSIM upon comparing a batch of generated images via this model to a batch of real images was approximately 0.67, where the closer the SSIM value to 1 the better. This model utilised 50 unique batches from the OASIS Brain training set. Please refer to the bottom of this README file for dependencies, how to run the model and further information regarding running this own your own system.

## Preprocessing Data for the DCGAN
The OASIS brain dataset is a set of preprocessed images, of which 9664 of these images are considered for training and 544 for testing. This model utilised 50 randomly chosen batches of size 16. Each image was resized to a resolution of 256x256, and then had their colour pallete changed to grayscale. They were then normalised to (-1, 1) for training within the DCGAN.


