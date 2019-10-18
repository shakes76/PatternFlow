## TensorFlow Implementation of skimage.exposure.adjust_sigmoid. 

This folder contains two python scripts. "**Contrast_Corrector_Source.py**" contains the source code for the 
contrast adjuster function **adjust_sigmoid** ported to TensorFlow (now called **Contrast_Sigmoid**). "**Contrast_Test_Script.py**" 
provides an example image (this can also be found in the ElaineSchenk folder as "**test_image.png**") and an implementation 
of the Constrast_Sigmoid function on this image. Specifically, it provides contrast adjustment for this image with the following parameters passed: **gain = 5**, **cutoff = 0.5**, and both the inverse and default sigmoid function applied. It then compares results to the original non-ported function (ajust_sigmoid). It can be observed that results agree. 

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





