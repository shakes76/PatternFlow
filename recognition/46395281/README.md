# StyleGAN implementation on OASIS dataset

COMP3710 Project  
Chun-Chiao Huang  
46395281  
  
## Model structue
### ProGAN strusture
![2b8c387b198eefedbeca6bd51f154ea](https://user-images.githubusercontent.com/64058748/139537018-7c6cac0c-9caf-4d2b-9a30-b0673936e46a.png)   
    Tero et al., 2018 
  
  The structure of styleGAN is similar to ProGAN, while the Generator is changed to the structure shown in the below figure.  
  The generator and the discriminator both start at 4 * 4 images and progressively grow to 256 * 256 images.
### Structure of Generator
![96097a522ee08b0721e0753d2355819](https://user-images.githubusercontent.com/64058748/139536816-1ef0d410-bed3-4b83-95f0-920c212481c0.png)   
   Tero et al., 2020    
   #### AdaIN
   Instead of random noise input to the generator, the input is now changed to constants.   
   A 8-layers mapping net is added to project random noise *z* to latent *w*.   
   Then, *w* go through a dense layer and become factors and biases for Adaptive Instance Normorlization. (AdaIN)
   #### Equlised Learning Rate (Weight Scaled)    
   All the weights of Conv layers and dense layers are devided by a factor (He's std) to ensure the dynamic range and the learning speed for them are the same. (Tero et al., 2018 )    
   #### Noise Injection   
   Noises are scaled by learnable parameters and then injected before every AdaIN.
   #### Fade-in
   During training, whenever the growing happens, the output of the previous layer will be upscaled and mixed with the current layer.
   The proportion of the current layer and the previous layer is controlled by *alpha*, which decays as the current layer being trained.
   #### Loss
   The loss used is Non-Saturating loss, suggested by tutor Siyu.
   No other loss is added as it works very well.
## Results    
### Images   
![brain](https://user-images.githubusercontent.com/64058748/139554285-448fd16b-0111-4bdd-84af-1741ebf9a78f.png)   
![brain](https://user-images.githubusercontent.com/64058748/139554323-e2f4a66d-ce2a-4e3e-be62-798d9f2b6e86.gif)   
 

### Loss
    

## Requirements    
## Usage  
### To train the model, run:
    python train.py [data_path] [results_path]  
where [data_path] is the path of dataset and the results_path is the path the results will be saved at.  
The results include trained denerator model, discriminator model, loss plot, images during training and a .gif image show the training process.
### To use the trained model to generate images, run:
    python generate.py [model_path] [results_path]   
where the [model_path] is the path of the trained generator and the [results_path] is the path the results will be saved at.  


## Reference
### Papers
  Karras, T., Laine, S., & Aila, T. (2020). A Style-Based Generator Architecture for Generative Adversarial Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, PP, 1–1. https://arxiv.org/pdf/1812.04948.pdf

  Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation. https://arxiv.org/pdf/1710.10196.pdf

### Codes   

https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan_test_discriminator.ipynb
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/StyleGAN
https://github.com/rosinality/style-based-gan-pytorch   
https://github.com/caffeinism/StyleGAN-pytorch
https://github.com/facebookresearch/pytorch_GAN_zoo

