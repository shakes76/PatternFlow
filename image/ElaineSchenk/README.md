## TensorFlow Implementation of skimage.exposure.adjust_sigmoid. 

This folder contains two python scripts. "**Contrast_Corrector_Source.py**" contains the source code for the 
contrast adjuster function **adjust_sigmoid** ported to TensorFlow (now called **Contrast_Sigmoid**). "**Contrast_Test_Script.py**" 
provides an example image (this can also be found in the ElaineSchenk folder as "**test_image.png**") and an implementation 
of the Constrast_Sigmoid function on this image. Specifically, it provides contrast adjustment for this image with the following parameters passed: **gain = 5**, **cutoff = 0.5**, and both the inverse and default sigmoid function applied. It then compares results to the original non-ported function (ajust_sigmoid). It can be observed that results agree. 

### Algorithmn Description:
The adjust_sigmoid function in the exposure module of skimage is an example of a contrast adjustment algorithmn for images. 
It can be applied to an array of pixel data to provide new pixel data, such that the reulting image has potentially better, and
more interpretable characteristics due to the contrast adjustment. This essentially works by normalising the pixel data and then stretching out the magnitudes about some cutoff point (the default being 0.5). The degree to which neighbouring intensities diverge
is parameterised by a gain coefficient. This and the cutoff coefficient are factors in a sigmoid function. This function is used in 
this application as it has the mathematical effect of "pulling away" neighbouring values. This naturally, provides the adjustment
of intensities required to alter contrast in an image. 
The inverse sigmoid function is also an option for this algorithmn. When using the inverse function, the dark aspects of an image are transformed to being the lighter and vice versa. 

**The test script in the ElaineSchenk Folder contains example images of the algorithm being applied to a relatively poor contrast
image**. 

### Sigmoid Function: 
Essentially the gain will effect how drastically the darkness/brightness between two pixels varies after adjustment. 
The cutoff will effect the point around which pixel values diverge after adjustment. Default is set to 0.5 (i.e. between
black and white in normalised image). 

> **Default Sigmoid**
> O = 1/(1+exp(gain*(cutoff-image/scale)))

> **Inverse Sigmoid**
> O = 1 - 1/(1+exp(gain*(cutoff-image/scale)))

### Comparing Parameters: 

> **adjust_sigmoid(image,cutoff = 0.5, gain = 10, inv = False)**
>
> image is an **ndarray**, cutoff and gain are sigmoid function parameters set to 0.5 and 10 default respectively. 
>
> inv is a bool argument, has default False. If set True, then applies inverse sigmoid. 
>
> output: returns an **ndarray**

> **Contrast_Sigmoid(image,cutoff = 0.5, gain = 10, inv = False)**
>
> image is an **ndarray**, cutoff and gain are sigmoid function parameters set to 0.5 and 10 default respectively. 
>
> Contrast_Sigmoid will convert the inputted image to an nd tensor. 
>
> inv is a bool argument, has default False. If set True, then applies inverse sigmoid. 
>
> output: returns an nd **tensor**. 





