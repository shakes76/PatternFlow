# Perceiver Transformer to Classify Alzheimer's Disease
### Zach Thomas 45287325
## Introduction
The task at hand was to classify images of brain scans as either Alzheimer's Disease (AD) or normal cognition (NC). This was attempted using a Perceiver transformer. Using transformer's to classify images is a fairly recent technique, with the main aim to use a generic model that makes few assumptions about the structure of the data. This is done with the intention of reducing bias in the model from assumptions (e.g. using convolutions makes the assumption the pixels near each other are related). 

The Perciever uses a technique surrounding the use of a latent array to reduce the time complexity of attention operations. Traditionally, the attention operations done on flattened images with size (M x C) are O(M^2), where M = height x width. The perciever utilises a latent array with dimension (N x D), where N << M. This reduces the complexity of attention operations to O(M^2). This allows for significantly deeper transformers to be built that still have a manageable time complexity for training. 

![Perceiver Architecture](https://user-images.githubusercontent.com/88659407/197125075-35594818-5ec6-4eef-833a-69ee1a3401ba.JPG)
[[1]](#1)

In order to solve the problem of abstracting away the image positions, the Perceiever uses positional encodings to allow the model to identify pixels that relate to each other. Two approaches were described in the paper. The first was to use Fourier based encodings and the second was to use a learnt encoding. This model opts for the second approach. 

## Results
The data was trained using Cross Entropy Loss and the Adam optimizer. Note that the original paper describes the use of the LAMB optimizer, However, as of the time of this project, this does not exist within the pytorch optimization tools. Due to memory constraints, a smaller model than described in the paper was used for training. It uses one cross attend cycle, with a latent transformer with depth 5. For the ImageNet classification in the paper, they use eight cross attend cycles with latent transformers of depth 8. 

Using a batch size of 5, a learning rate of 0.005 and 10 epochs, the following results were obtained. 

## Project Structure
The code for loading the data is contained in **`dataset.py`**. The methods ``` train_data_loader``` and ``` test_data_loader`` both take a directory and a batch size as arguments. The directory must contain two folders, one names train and one named test. Within both of these, there needs to be two folders, one named AD and one named MC. These contain the respective images for both classes of brain scans. 

In order to train the data, the script **`train.py`** should be run. This script contains various constants that the Perceiever takes as parameters. This includes the latent dimensions N and D as well as parameters for the depth of the Perceiever. It also includes training constants such as the batch size, learning rate and the number of epochs. This script saves the model in the path specified by MODEL_PATH. This should be altered as desired. 

## References
<a id="1">[1]</a>
Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021, July). Perceiver: General perception with iterative attention. In International conference on machine learning (pp. 4651-4664). PMLR.
