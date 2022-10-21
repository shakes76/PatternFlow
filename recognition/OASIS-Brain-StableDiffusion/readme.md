# OASIS Brain MRI Stable Diffusion

##### Jonathan Allen (s4640736)

## Model Description
Stable Diffusion is a deep learning image generation model. Unlike other diffusion models that denoise in pixel space, Stable Diffusion uses a Latent Diffusion Model (LDM) to add and remove noise in the latent space, making it more time and space efficient. An Encoder is used to transform images into the latent space, while a Decoder is used to transform the latent space back into pixel space. In the model, each denoising step is performed by a modified U-Net architecture consisting of cross attention and positional embedding. The cross attention increases the segmentation accuracy by enhancing prominent regions of the image and suppressing irrelevant regions. The positional embedding allows the model to know what noising or denoising step it is on, vital to add and remove the correct amount of noise to an image, as the basic U-Net architecture does not allow for this. For more information on the model used and for the mathematics behind it please read this [paper](https://arxiv.org/pdf/2112.10752.pdf).

![image](https://miro.medium.com/max/1400/0*rW_y1kjruoT9BSO0.png)

Source: https://arxiv.org/pdf/2112.10752.pdf

For the model implemented here, the conditioning part of the Stable Diffusion Model was intentionally dismissed as unnecessary since the task is to only recreate one specific type of data set. The conditioning would be needed if we were trying to recreate many types of data from a given input. For example, creating an image from text, in this case, text is the conditioning element.

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

![Unet Diagram](https://miro.medium.com/max/1200/1*f7YOaE4TWubwaFF7Z1fzNw.png)

Source: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

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

## Training `train.py`
The training loop for this network is a standard PyTorch training loop consisting of the main model, optimizer, and loss function.
### Main Model
The main model for this training loop is the U-Net outlined above in the modules section.
### Optimizer
The optimizer chosen for this training loop was the same one mentioned in the paper due to its generalize well. This optimizer is Adam from `torch.optim.adam`.
### Loss Function
The loss function chosen for this training loop is Mean Squared Error (squared L2 norm). This loss function was chosen since the network is extremely connected and noisy, it provides some generalization and smoothing. The loss function can be found at `torch.nn.MSELoss`.
### Training Loop
```python
for  epoch  in  range(epochs):
	epoch_loss  =  0
	for  idx, (data, _) in  enumerate(tqdm(train_dataloader)):
		data  =  data.to(device)
		position  =  get_sample_pos(data.shape[0]).to(device)
		noisy_x, noise  =  add_noise(data, position)
		predicted_noise  =  model(noisy_x, position)
		loss  =  loss_fn(noise, predicted_noise)
		optimizer.zero_grad()
		loss.backward()
		epoch_loss  +=  loss.item()
		optimizer.step()
	tracked_loss.append([epoch_loss  /  dataloader_length])
	print("Current Loss ==> {}".format(epoch_loss/dataloader_length))
	test_loss.append(test_model(model, test_path, batch_size, device))
```
The last line of the training loop calls the validation loop for that epoch to gather data on the test data to ensure the mode is not over-fitting. This is then saved to a csv and analyzed with Microsoft Excel.

### Validation Loop
```python
for  idx, (data, _) in  enumerate(tqdm(test_dataloader)):
	data  =  data.to(device)
	position  =  get_sample_pos(data.shape[0]).to(device)
	noisy_x, noise  =  add_noise(data, position)
	predicted_noise  =  model(noisy_x, position)
	loss  =  loss_fn(noise, predicted_noise)
	running_loss  +=  loss.item()
```
### Training Results
#### Graph showing the running loss of the training set in the training loop
![Training Loss Vs Epochs](https://lh3.googleusercontent.com/pw/AL9nZEXH2I2U1lkr2GSYLzaDYCpROtEi_1OBWQLEEBIUu50t-2Rl5OBAeSYB05HEHLiOlItM_UJbGPEldyLEzeI_46pKSp8fuvKqYB1iA5NfXHwDZUqOyJlrYPMAtXYspKBMeeKLyjV9KHgCMXu5Rpgl3aCJ=w752-h452-no?authuser=0)
#### Graph showing the running loss of the test set in the validation loop
![Testing Loss vs Epoch](https://lh3.googleusercontent.com/pw/AL9nZEUevH6b6bYM2b4t073QSLS3iTE9O2KyasB9qhwcNqdSRcER5fsRassBdCob0oDd1uuZ7WHMSpzEigIQY1Jd_HyiAT6pnKFMu_tLvZwFHt_XvkD1ZTRspbIA4_cU-ci_1FW0_52kIis50unYYOygVD8X=w751-h452-no?authuser=0)
Showing only the first 30 Epochs as the loss flattens off and platues for the rest of the epochs.
### Training Your Own Model
To train your own stable diffusion model. Ensure the hyperparameters meet your specification in `main()` in `train.py` (below is the default example one).
```python 
def  main():
	#hyperparameters
	device  =  "cuda"
	lr  =  3e-4
	train_path  =  r".\OASIS-Brain-Data\training_data"
	test_path  =  r".\OASIS-Brain-Data\test_data"
	model  =  UNet().to(device)
	batch_size  =  12
	epochs  =  200
```
Then run `train.py` in the terminal with the command `python train.py`.

## Results `predict.py`
Once a model is trained, `predict.py` can be used to load that model using `load_model()`.
After the model is loaded, it can be used to generate images from noise using the functions `show_single_image()` and `show_x_images()`. The path to that model must be specified in the `main` method. By default `predict.py` generates 6 images in a row and saves it.

### Images Generated From Stable Diffusion Model
Here are six Brain MRIs generated from this stable diffusion model

![Generated Brain Images](https://lh3.googleusercontent.com/pw/AL9nZEWOVw8GZ23W_Nn1bzlelZnFbdSNY1OtXRLf2EWPLUAf0EIg8Naw0rXTSUadluIzml-r91DJxK2BNYnNodYBzRSVtL9RLReduJayQ2dP9kGNSFuXGuFsG1LBycxD38to4LS8jNWgJiY5FmKTNbk08SpY=w398-h68-no?authuser=0 =796x136)
To generate your own images, just run `python predict.py`in the terminal. Remember a model has to be trained, saved, and then its path has to be referenced appropriately in `predict.py`.

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)


## Dependencies
```
# Name                    Version                   Build  Channel
absl-py                   1.2.0              pyhd8ed1ab_0    conda-forge
aiohttp                   3.8.1           py310he2412df_1    conda-forge
aiosignal                 1.2.0              pyhd8ed1ab_0    conda-forge
argon2-cffi               21.3.0             pyhd8ed1ab_0    conda-forge
argon2-cffi-bindings      21.2.0          py310he2412df_2    conda-forge
asttokens                 2.0.8              pyhd8ed1ab_0    conda-forge
async-timeout             4.0.2              pyhd8ed1ab_0    conda-forge
attrs                     22.1.0             pyh71513ae_1    conda-forge
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                        py_2    conda-forge
backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
beautifulsoup4            4.11.1             pyha770c72_0    conda-forge
blas                      2.116                       mkl    conda-forge
blas-devel                3.9.0              16_win64_mkl    conda-forge
bleach                    5.0.1              pyhd8ed1ab_0    conda-forge
blinker                   1.4                        py_1    conda-forge
brotli                    1.0.9                h8ffe710_7    conda-forge
brotli-bin                1.0.9                h8ffe710_7    conda-forge
brotlipy                  0.7.0           py310he2412df_1004    conda-forge
bzip2                     1.0.8                h8ffe710_4    conda-forge
c-ares                    1.18.1               h8ffe710_0    conda-forge
ca-certificates           2022.9.24            h5b45459_0    conda-forge
cachetools                5.2.0              pyhd8ed1ab_0    conda-forge
certifi                   2022.9.24          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1          py310hcbf9ad4_0    conda-forge
charset-normalizer        2.1.1              pyhd8ed1ab_0    conda-forge
click                     8.1.3           py310h5588dad_0    conda-forge
colorama                  0.4.5              pyhd8ed1ab_0    conda-forge
cryptography              37.0.1          py310h21b164f_0
cudatoolkit               11.6.0              hc0ea762_10    conda-forge
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
debugpy                   1.6.3           py310h8a704f9_0    conda-forge
decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
defusedxml                0.7.1              pyhd8ed1ab_0    conda-forge
entrypoints               0.4                pyhd8ed1ab_0    conda-forge
executing                 1.0.0              pyhd8ed1ab_0    conda-forge
flit-core                 3.7.1              pyhd8ed1ab_0    conda-forge
fonttools                 4.37.1          py310he2412df_0    conda-forge
freetype                  2.12.1               h546665d_0    conda-forge
frozenlist                1.3.1           py310he2412df_0    conda-forge
gettext                   0.19.8.1          ha2e2712_1008    conda-forge
glib                      2.72.1               h7755175_0    conda-forge
glib-tools                2.72.1               h7755175_0    conda-forge
google-auth               2.11.0             pyh6c4a22f_0    conda-forge
google-auth-oauthlib      0.4.6              pyhd8ed1ab_0    conda-forge
grpc-cpp                  1.48.1               h535cfc9_1    conda-forge
grpcio                    1.48.1          py310hd8b4215_1    conda-forge
gst-plugins-base          1.20.3               h001b923_1    conda-forge
gstreamer                 1.20.3               h6b5321d_1    conda-forge
icu                       70.1                 h0e60522_0    conda-forge
idna                      3.3                pyhd8ed1ab_0    conda-forge
importlib-metadata        4.11.4          py310h5588dad_0    conda-forge
importlib_resources       5.9.0              pyhd8ed1ab_0    conda-forge
intel-openmp              2022.1.0          h57928b3_3787    conda-forge
ipykernel                 6.15.2             pyh025b116_0    conda-forge
ipython                   8.5.0              pyh08f2357_1    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
jedi                      0.18.1             pyhd8ed1ab_2    conda-forge
jinja2                    3.1.2              pyhd8ed1ab_1    conda-forge
joblib                    1.1.0              pyhd8ed1ab_0    conda-forge
jpeg                      9e                   h8ffe710_2    conda-forge
jsonschema                4.15.0             pyhd8ed1ab_0    conda-forge
jupyter_client            7.3.5              pyhd8ed1ab_0    conda-forge
jupyter_contrib_core      0.4.0              pyhd8ed1ab_0    conda-forge
jupyter_contrib_nbextensions 0.5.1              pyhd8ed1ab_2    conda-forge
jupyter_core              4.11.1          py310h5588dad_0    conda-forge
jupyter_highlight_selected_word 0.2.0           py310h5588dad_1005    conda-forge
jupyter_latex_envs        1.4.6           pyhd8ed1ab_1002    conda-forge
jupyter_nbextensions_configurator 0.4.1              pyhd8ed1ab_2    conda-forge
jupyterlab_pygments       0.2.2              pyhd8ed1ab_0    conda-forge
kiwisolver                1.4.4           py310h476a331_0    conda-forge
krb5                      1.19.3               h1176d77_0    conda-forge
lcms2                     2.12                 h2a16943_0    conda-forge
lerc                      4.0.0                h63175ca_0    conda-forge
libabseil                 20220623.0      cxx17_h1a56200_4    conda-forge
libblas                   3.9.0              16_win64_mkl    conda-forge
libbrotlicommon           1.0.9                h8ffe710_7    conda-forge
libbrotlidec              1.0.9                h8ffe710_7    conda-forge
libbrotlienc              1.0.9                h8ffe710_7    conda-forge
libcblas                  3.9.0              16_win64_mkl    conda-forge
libclang                  14.0.6          default_h77d9078_0    conda-forge
libclang13                14.0.6          default_h77d9078_0    conda-forge
libdeflate                1.13                 h8ffe710_0    conda-forge
libffi                    3.4.2                h8ffe710_5    conda-forge
libglib                   2.72.1               h3be07f2_0    conda-forge
libiconv                  1.16                 he774522_0    conda-forge
liblapack                 3.9.0              16_win64_mkl    conda-forge
liblapacke                3.9.0              16_win64_mkl    conda-forge
libogg                    1.3.4                h8ffe710_1    conda-forge
libpng                    1.6.37               h1d00b33_4    conda-forge
libprotobuf               3.21.5               h12be248_3    conda-forge
libsodium                 1.0.18               h8d14728_1    conda-forge
libsqlite                 3.39.3               hcfcfb64_0    conda-forge
libtiff                   4.4.0                h92677e6_3    conda-forge
libuv                     1.44.2               h8ffe710_0    conda-forge
libvorbis                 1.3.7                h0e60522_0    conda-forge
libwebp-base              1.2.4                h8ffe710_0    conda-forge
libxcb                    1.13              hcd874cb_1004    conda-forge
libxml2                   2.9.14               hf5bbc77_4    conda-forge
libxslt                   1.1.35               h34f844d_0    conda-forge
libzlib                   1.2.12               h8ffe710_2    conda-forge
lxml                      4.9.1           py310he2412df_0    conda-forge
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
markdown                  3.4.1              pyhd8ed1ab_0    conda-forge
markupsafe                2.1.1           py310he2412df_1    conda-forge
matplotlib                3.5.3           py310h5588dad_2    conda-forge
matplotlib-base           3.5.3           py310h7329aa0_2    conda-forge
matplotlib-inline         0.1.6              pyhd8ed1ab_0    conda-forge
mistune                   2.0.4              pyhd8ed1ab_0    conda-forge
mkl                       2022.1.0           h6a75c08_874    conda-forge
mkl-devel                 2022.1.0           h57928b3_875    conda-forge
mkl-include               2022.1.0           h6a75c08_874    conda-forge
msys2-conda-epoch         20160418                      1    conda-forge
multidict                 6.0.2           py310he2412df_1    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
nb_conda_kernels          2.3.1           py310h5588dad_1    conda-forge
nbclient                  0.6.7              pyhd8ed1ab_0    conda-forge
nbconvert                 7.0.0              pyhd8ed1ab_0    conda-forge
nbconvert-core            7.0.0              pyhd8ed1ab_0    conda-forge
nbconvert-pandoc          7.0.0              pyhd8ed1ab_0    conda-forge
nbformat                  5.4.0              pyhd8ed1ab_0    conda-forge
nest-asyncio              1.5.5              pyhd8ed1ab_0    conda-forge
notebook                  6.4.12             pyha770c72_0    conda-forge
numpy                     1.23.2          py310h8a5b91a_0    conda-forge
oauthlib                  3.2.1              pyhd8ed1ab_0    conda-forge
openjpeg                  2.5.0                hc9384bd_1    conda-forge
openssl                   1.1.1q               h8ffe710_0    conda-forge
packaging                 21.3               pyhd8ed1ab_0    conda-forge
pandas                    1.4.4                    pypi_0    pypi
pandoc                    2.19.2               h57928b3_0    conda-forge
pandocfilters             1.5.0              pyhd8ed1ab_0    conda-forge
parso                     0.8.3              pyhd8ed1ab_0    conda-forge
pathlib                   1.0.1           py310h5588dad_6    conda-forge
pcre                      8.45                 h0e60522_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.2.0           py310h52929f7_2    conda-forge
pip                       22.2.2             pyhd8ed1ab_0    conda-forge
pkgutil-resolve-name      1.3.10             pyhd8ed1ab_0    conda-forge
ply                       3.11                       py_1    conda-forge
prometheus_client         0.14.1             pyhd8ed1ab_0    conda-forge
prompt-toolkit            3.0.31             pyha770c72_0    conda-forge
protobuf                  3.19.5                   pypi_0    pypi
psutil                    5.9.2           py310h8d17308_0    conda-forge
pthread-stubs             0.4               hcd874cb_1001    conda-forge
pure_eval                 0.2.2              pyhd8ed1ab_0    conda-forge
pyasn1                    0.4.8                      py_0    conda-forge
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.21               pyhd8ed1ab_0    conda-forge
pygments                  2.13.0             pyhd8ed1ab_0    conda-forge
pyjwt                     2.4.0              pyhd8ed1ab_0    conda-forge
pyopenssl                 22.0.0             pyhd8ed1ab_0    conda-forge
pyparsing                 3.0.9              pyhd8ed1ab_0    conda-forge
pyqt                      5.15.7          py310hbabf5d4_0    conda-forge
pyqt5-sip                 12.11.0         py310h8a704f9_0    conda-forge
pyrsistent                0.18.1          py310he2412df_1    conda-forge
pysocks                   1.7.1              pyh0701188_6    conda-forge
python                    3.10.6          h9a09f29_0_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python-fastjsonschema     2.16.1             pyhd8ed1ab_0    conda-forge
python_abi                3.10                    2_cp310    conda-forge
pytorch                   1.12.1          py3.10_cuda11.6_cudnn8_0    pytorch
pytorch-model-summary     0.1.1                      py_0    conda-forge
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.2.1                 pypi_0    pypi
pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
pywin32                   303             py310he2412df_0    conda-forge
pywinpty                  2.0.7           py310h00ffb61_0    conda-forge
pyyaml                    6.0             py310he2412df_4    conda-forge
pyzmq                     23.2.1          py310h73ada01_0    conda-forge
qt-main                   5.15.4               h467ea89_2    conda-forge
re2                       2022.06.01           h0e60522_0    conda-forge
requests                  2.28.1             pyhd8ed1ab_1    conda-forge
requests-oauthlib         1.3.1              pyhd8ed1ab_0    conda-forge
rsa                       4.9                pyhd8ed1ab_0    conda-forge
scikit-learn              1.1.2           py310h3a564e9_0    conda-forge
scipy                     1.9.1           py310h578b7cb_0    conda-forge
send2trash                1.8.0              pyhd8ed1ab_0    conda-forge
setuptools                65.3.0             pyhd8ed1ab_1    conda-forge
sip                       6.6.2           py310h8a704f9_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
soupsieve                 2.3.2.post1        pyhd8ed1ab_0    conda-forge
sqlite                    3.39.3               hcfcfb64_0    conda-forge
stack_data                0.5.0              pyhd8ed1ab_0    conda-forge
tbb                       2021.5.0             h91493d7_2    conda-forge
tensorboard               2.10.1             pyhd8ed1ab_0    conda-forge
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1              pyhd8ed1ab_0    conda-forge
terminado                 0.15.0          py310h5588dad_0    conda-forge
threadpoolctl             3.1.0              pyh8a188c0_0    conda-forge
tinycss2                  1.1.1              pyhd8ed1ab_0    conda-forge
tk                        8.6.12               h8ffe710_0    conda-forge
toml                      0.10.2             pyhd8ed1ab_0    conda-forge
torch-summary             1.4.5                    pypi_0    pypi
torch-tb-profiler         0.4.0                    pypi_0    pypi
torchaudio                0.12.1              py310_cu116    pytorch
torchvision               0.13.1              py310_cu116    pytorch
tornado                   6.2             py310he2412df_0    conda-forge
tqdm                      4.64.1             pyhd8ed1ab_0    conda-forge
traitlets                 5.3.0              pyhd8ed1ab_0    conda-forge
typing-extensions         4.3.0                hd8ed1ab_0    conda-forge
typing_extensions         4.3.0              pyha770c72_0    conda-forge
tzdata                    2022c                h191b570_0    conda-forge
ucrt                      10.0.20348.0         h57928b3_0    conda-forge
unicodedata2              14.0.0          py310he2412df_1    conda-forge
urllib3                   1.26.11            pyhd8ed1ab_0    conda-forge
vc                        14.2                 hb210afc_7    conda-forge
vs2015_runtime            14.29.30139          h890b9b1_7    conda-forge
wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
webencodings              0.5.1                      py_1    conda-forge
werkzeug                  2.2.2              pyhd8ed1ab_0    conda-forge
wheel                     0.37.1             pyhd8ed1ab_0    conda-forge
win_inet_pton             1.1.0           py310h5588dad_4    conda-forge
winpty                    0.4.3                         4    conda-forge
xorg-libxau               1.0.9                hcd874cb_0    conda-forge
xorg-libxdmcp             1.1.3                hcd874cb_0    conda-forge
xz                        5.2.6                h8d14728_0    conda-forge
yaml                      0.2.5                h8ffe710_2    conda-forge
yarl                      1.7.2           py310he2412df_2    conda-forge
zeromq                    4.3.4                h0e60522_1    conda-forge
zipp                      3.8.1              pyhd8ed1ab_0    conda-forge
zlib                      1.2.12               h8ffe710_2    conda-forge
zstd                      1.5.2                h7755175_4    conda-forge
```
