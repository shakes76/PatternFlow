# OASIS Brain MRI Stable Diffusion

##### Jonathan Allen (s4640736)

## Model Description
Stable Diffusion is a deep learning image generation model. Unlike other diffusion models that denoise in pixel space, Stable Diffusion uses a Latent Diffusion Model (LDM) to add and remove noise in the latent space, making it more time and space efficient. An Encoder is used to transform images into the latent space, while a Decoder is used to transform the latent space back into pixel space. In the model, each denoising step is performed by a modified U-Net architecture consisting of cross attention and positional embedding. The cross attention increases the segmentation accuracy by enhancing prominent regions of the image and suppressing irrelevant regions. The positional embedding allows the model to know what noising or denoising step it is on, vital to add and remove the correct amount of noise to an image, as the basic U-Net architecture does not allow for this. For more information on the model used and for the mathematics behind it please read this [paper](https://arxiv.org/pdf/2112.10752.pdf).

![image](https://miro.medium.com/max/1400/0*rW_y1kjruoT9BSO0.png)

Source: https://arxiv.org/pdf/2112.10752.pdf

For the model implemented here, the conditioning part of the stable diffusion model was intentionally dismissed as unnecessary since the task is to only recreate one specific type of data set. The conditioning would be needed if we were trying to recreate many types of data from a given input. For example, creating an image from text, in this case, text is the conditioning element.

## How Stable Diffusion Works
At a high level, diffusion models work by progressively altering the training data by adding noise, and then "learn" to recover the data by reversing this noising process. After training, this reversing of the noising process can be applied to random noise, hence generating new data. In other words, a complete diffusion model can produce coherent images from initial random noise. 

In this case specifically, the model is trained on OASIS brain MRI data, it adds a specific amount of noise to the image, and then iteratively learns how to remove the added noise in hopes to recover the altered training image. Once trained, it can be given a similar photo containing nothing but random noise, and then using the learned denoising process, generate a photo-realistic image resembling the training data.

This image shows a visual representation of the model noising an image and then learning to denoise it by reversing the process. Noising of the image going left to right on the first row and denoising of the image going right to left on the second row.
![Example Noising Process](https://lh3.googleusercontent.com/432gw-wUaTSikRtRp2IjoIRxM_xLYhy06LXcUYfHmVZoGJfWGl88HX5DO4jUxxhaZdPY_yDsKymTyHqO3oNz5vVv71poNJAwkbaYXtStpA5XyjPTqjvA3NNJK5rJndkgru4f9DPfqdqwKQuazuND-yWpn0uplZ-6mUfboiLh1BNEu1a92Pxm83gDtYfhr7chxzZW1ibgPp6dJ8G75yWy26SxjA6n9hgSDpqQgQj-QmRZURf7zcXnGbPMvk_1Je-uB2nzxIfswWVyb7isxdBKU75NzyV-a6zNLdZY9CDEgU50jzrCYeAA8_mjWNFDHsG_kyQgsCbAcdt4Logvk-d-ipqi12LRE83XsfOWopI9-Bs9FDN0eDBndNTPWh_PsGzaw1ZyAn-tJSzmtRjz3DQnnQ3J34BvFiYkZyPSBErDLvAYemeIphUZ-u7qxlbgi9HmkOU_g4AtMEc637LuMhD8bQN8u9y2cA74giWEce_Xw8E62oR4oowKkKCWWLw6HFs_JoLAAb4NJ6eJs_2JDOvDcKVVyNt07_mWZdNx2xvB2bjEoKIf-s4iBMT0q0RcxqUfhZk8ItM9nRuEkrx1DuGc1BuDWLjsfSUIZ5UHRgRlO11G6-zHhmvPUyAYnguS3k6bs8rTrMmGf6Fu6zWIydvxEUtsfJ97ZsRbmDCe1pbq4dVF-PMLoeTAKQagh0iTd6gvlsHijNsq2erqU0tSiMyVlGOk8tsZs5hVlFDJXCxaMXQpi6Mbpb_ErI-azmB0-CUi8mAdOphz2AKSp_0dMTgyIn25Gc3JI8BFerIVYSee2zjMYPb9NxNskS77yNRyPNCMWKAu4Ogv4zQihrPltHwo0kvz82Fcz6_XjRBy3NOh6NvyRBNujKz24_90iKvrg8wxNo6l4v5Z93MXhv70ctW3d8QPR1zL_I145aBp1A=w642-h319-no?authuser=0)
## Project Structure
This project is built using Python3.10.6 and PyTorch1.12.1.
## Data Pipeline `dataset.py`
### Data Loading
The OASIS Brain MRI dataset is the data this model was trained against. 
This data can be found [here](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA). 
Example MRI slice from the dataset
![Brain MRI](https://lh3.googleusercontent.com/pw/AL9nZEViraVfAx4nNjNFk7ga3r2QBN5zKUvgXMg7C-OvQLNKJN_mnTjKSrS4PmHYn5VZlt0ZUenfr15Bym4h08bWUF6XhivR0WwOXxGN1IJM2C7_oxYpSskmnNR9tzFdSVWPNmuhdTFF24qV4DDC4qrnkUx2=s256-no?authuser=0)

Source: [OASIS Brain MRI dataset](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA)

To load this data into the program, I used a function called `load_dataset` found in `dataset.py`. This function loads data from a folder, normalizes it, and puts it in a pytorch dataloader for ease of use.
Two completely separate folders were used to store the training data and test validation data, `training_data` and `test_data`. The `training_data` consisted of 8,406 images, while `test_data` contains 544 images. Separating this data into two different folders ensured that the model never saw the test data until it was validated.
Although the data looks grayscale, I did not transform the data into grayscale for training in hopes to keep the model open to train coloured images in the future.

### Preprocessing/Normalization
All the input data needs to be preprocessed and normalized to be useful for the model. This normalization is also done in the `load_dataset` function found in `dataset.py`. This is done using `torchvision`and consists of four lines.
The first line resizes the image to be a quarter size larger to allow for random cropping
`torchvision.transforms.Resize(image_size+round(0.25*image_size))`
The second line randomly crops the image back down to its intended size but slightly off center
`torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))`
The third line converts it to a PyTorch tensor
`torchvision.transforms.ToTensor()`
The fourth line normalizes it using the mean and std
`torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
This preprocessing and normalization is in place to ensure the model receives data to generalize not just remember. 

## Modules `modules.py`
As discussed above a modified U-Net was used to denoise the data in latent space. This modified U-Net has cross attention blocks along with positional embedding and will discussed later.

A general U-Net is an encoder and decoder convolutional network with skip connections between each opposing block. It was originally created to produce image segmentation mapping which makes it the golden choice to use in a stable diffusion model.

![Unet Diagram](https://photos.google.com/share/AF1QipMt0Y7G1djhopVX_kNX0DHn0OMqmg_NmPQbAZJGKRk0BaFRx0LoWFozJttkpl-msw/photo/AF1QipM5fX5cAPG8LiJQhOXhwFXaM9USl6jgzIiqCUY?key=Z1RWSjJDNld1N3pCV1BnZk1IMGNXbkd1dUZpempB)
Source: https://www.researchgate.net/figure/The-architecture-of-Unet_fig2_334287825

The U-Net implemented for this model is slightly different and consists of the following blocks.
- `ConvReluBlock` - object consisting of a double convolution rectified linear layer and a group normalization
- `EncoderBlock` - block consisting of a max pooling layer followed by 2 ConvReluBlocks
concatenated with the embedded position tensor
- `DecoderBlock` - Decoder block consisting of an upsample layer followed by 2 ConvReluBlocks
concatenated with the embedded position tensor
- `AttentionBlock` - Transformer attention block to enhance some parts of the data and diminish other parts
- `UNet` - Unet model consisting of a decoding block, an encoding block, cross attention, and residual skip connections along with positional encoding

These blocks used to implement the network can be found in the `modules.py` file.
To better understand the model, below are annotated forward steps for each block.
### ConvReluBlock
```python
# Block 1
x1  =  self.conv1(x) 		#nn.Conv2d()
x2  =  self.gNorm(x1) 		#nn.GroupNorm()
x3  =  self.relu(x2) 		#nn.ReLU()
# Block 2
x4  =  self.conv2(x3) 		#nn.Conv2d()
x5  =  self.gNorm(x4) 		#nn.GroupNorm()
# Handle Residuals
if (self.residual_connection):
	x6  =  F.relu(x  +  x5)
else:
	x6  =  x5
return  x6
```
### EncoderBlock
```python
x  =  self.pool(x) 		#nn.MaxPool2d()
x  =  self.encoder_block1(x) 	#ConvReluBlock()
x  =  self.encoder_block2(x) 	#ConvReluBlock()
emb_x  =  self.embedded_block(position)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) #nn.ReLU() followed by nn.Linear()
return  x  +  emb_x 		#positional embedding, emb_x, is added in at every step
```

### DecoderBlock
```python
x  =  self.upSample_block(x) 			#nn.Upsample()
x  =  torch.cat([skip_tensor, x], dim=1) 	#Adding in the skip connections from encoder
x  =  self.decoder_block1(x) 			#ConvReluBlock()
x  =  self.decoder_block2(x) 			#ConvReluBlock()
emb_x  =  self.embedded_block(position)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) #nn.ReLU() followed by nn.Linear()
return  emb_x  +  x 				#positional embedding, emb_x, is added in at every step
```
### UNet
```python
position  =  position.unsqueeze(-1).type(torch.float)
position  =  self.positional_embedding(position, self.pos_dim)
# Encoder forward step							# in --> out (Tensor Size)
x1  =  self.in_layer(x) 			#ConvReluBlock() 	# 3 --> 64
x2  =  self.encoder1(x1, position) 		#EncoderBlock()		# 64 --> 128
x2  =  self.attention1(x2) 			#AttentionBlock()	# 128 --> 32
x3  =  self.encoder2(x2, position) 		#EncoderBlock()		# 128 --> 256
x3  =  self.attention2(x3)			#AttentionBlock()	# 256 --> 16
x4  =  self.encoder3(x3, position) 		#EncoderBlock()		# 256 --> 256
x4  =  self.attention3(x4) 			#AttentionBlock()	# 256 --> 8
 
# Bottle neck forward step						# in --> out (Tensor Size)
x4  =  self.b1(x4) 				#ConvReluBlock()	# 256 --> 512
x4  =  self.b2(x4) 				#ConvReluBlock()	# 512 --> 512
x4  =  self.b3(x4) 				#ConvReluBlock()	# 512 --> 256

# Decoder forward step							# in --> out (Tensor Size)
x  =  self.decoder1(x4, x3, position)		#DecoderBlock() 	# 512 --> 128
x  =  self.attention4(x) 			#AttentionBlock()	# 128 --> 16
x  =  self.decoder2(x, x2, position) 		#DecoderBlock()		# 256 --> 64
x  =  self.attention5(x) 			#AttentionBlock()	# 64 --> 32
x  =  self.decoder3(x, x1, position) 		#DecoderBlock()		# 128 --> 64
x  =  self.attention6(x) 			#AttentionBlock()	# 64 --> 64
out  =  self.out_layer(x) 			#nn.Conv2d()		# 64 --> 3

return  out
```
