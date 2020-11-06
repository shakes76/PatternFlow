# OASIS Brain Dataset Generative DCGAN Model 
This model aims to generate accurate brain scans through the utilisation of a DCGAN trained on the OASIS Brain dataset. The completed model  results in a "reasonably clear image" and a Structured Similarity (SSIM) of approximately 0.65 when comparing a batch of iamges produced by the generator with 50 different batches of images from the training set.

## OASIS Dataset Processing
The OASIS brain dataset is preprocessed, consisting of 9,664 brain scans for training purposes and 544 for testing. 

The images in the dataset were converted to gray-scale, clipped to a resolution of 256x256, and normalised to (-1,1) for the purposes of training. A batch size of 16 was used to train the model, and 50 batches were randomly chosen from the training set in order to generate an SSIM value for every epoch the model was ran.