import tensorflow as tf
import matplotlib.pyplot as plt
  
def image_histogram(IMAGE_PATH):
  image_array = plt.imread(IMAGE_PATH) 
  print(image_array.shape)
  plt.imshow(image_array) 
  plt.show()
  plt.xlabel('Reference Image')
  print(type(image_array[0][0]))
  image_shape_original = list(image_array.shape) 
  print(image_shape_original) 
  image_flattened_shape = [image_shape_original[0]*image_shape_original[1],1]
