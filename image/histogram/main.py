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

  image_placeholder = tf.sort(image_placeholder)
  count = tf.math.bincount(image_placeholder)
  init = tf.global_variables_initializer()
  S = tf.Session()
  S.run(init)
  fd = {image_placeholder:image_array}
  count = S.run(count,feed_dict = fd)
  int_list = []
  for i in range(0,len(count)):
    int_list.append(i)
    
    
  plt.bar(int_list,count)
  plt.xlabel('image_intensities')
  plt.ylabel('intensity-frequencies')
  plt.show()

  
image_histogram('./lighthouse.jpg')
