# L0 Gradient Smoothing

L0 gradient smoothing is a way to smooth images by limiting the number of non-zero gradients. It stands out 
from other smoothing methods thanks to its edge preserving properties. 

## Dependencies
Tensorflow >= 2.0.0

## Example Usage

1. Load the desired data into either a numpy array or Tensor
    * An arbitrary number of channels is supported
    * The last dimension should be the channels
    * The data should be scaled to be within the range [0, 1]
    
    ```python
    import numpy as np
    from PIL import Image
    
    image = Image.open('path/to/image/file')
    image_data = np.array(image) / 255.
    ```

2. Pass the image data to the l0_gradient_smoothing function
    * You can also specify additional parameters such as the smoothing factor
    * See the function docstring for information on what each parameter does

    ```python
    smoothed_result = l0_gradient_smoothing(image_data, smoothing_factor=0.015)
    ```
    
### Example Output
Input:

![](resources/wonka.png)

Output (default parameters):

![](resources/wonka_result.png)

Output (smoothing_factor = 0.1):

![](resources/wonka_result2.png)
     
# References

[1] Xu, L., Lu, C., Xu, Y. and Jia, J. (2011). Image smoothing via L0 gradient minimization. *Proceedings of the 2011 SIGGRAPH Asia Conference on - SA '11.*