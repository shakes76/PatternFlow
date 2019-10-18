# Adjust Sigmoid

The adjust_sigmoid function performs sigmoid correction on the inputted image. The function adjusts the value of each of the pixels according to the algorithm Output = 1 / (1 + exp(gain * (cutoff - Input'))).  Input' = Input / range, where range = (max value of data type) - (min value of data type)

# Example

Below is an example of how the module can be used to perform sigmoid correction on a real image.

```python
import matplotlib.pyplot as plt
from skimage import data
from adjust_sigmoid import adjust_sigmoid

# Get the moon image from skimage
image = data.moon()

# Adjust the image with the sigmoid correction
adjusted = adjust_sigmoid(image)

# Plot the original and adjusted for comparision
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title("Original")
fig.add_subplot(1, 2, 2)
plt.imshow(adjusted, cmap=plt.cm.gray)
plt.title("Adjusted")
plt.show()
```
#Comparision
![Comparision](example.jpg)
