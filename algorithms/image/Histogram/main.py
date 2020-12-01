import tensorflow as tf
import matplotlib.pyplot as plt

def image_histogram(IMAGE_PATH): # input the path of the image
  image_array = plt.imread(IMAGE_PATH) # image read, creating a matrix containing pixel intensities 
  print(image_array.shape) # print image shape
  plt.imshow(image_array)
  plt.show()
  plt.xlabel('Reference Image') # labelling
  print(type(image_array[0][0])) # type of image intensity, with first 0 being the row and the other 0 is the element 
  image_shape_original = list(image_array.shape) #creating list because we cant use tuple  
  print(image_shape_original)
  image_flattened_shape = [image_shape_original[0]*image_shape_original[1],1] # flattened image, making it 1-d vector, (rows*column, 1) output 
  image_placeholder = tf.placeholder(tf.int32,image_shape_original) # placeholder for image

  count = tf.math.bincount(image_placeholder) # pixel intensities, giving frequencies of each PI 
  init = tf.global_variables_initializer() # initiating session 
  S = tf.Session() 
  S.run(init)
  fd = {image_placeholder:image_array} # creating feed dict
  count = S.run(count,feed_dict = fd) # extracting count on the image array
  int_list = []
  for i in range(0,len(count)): # 0-255 list, for all frequency we are getting the unique intensity values 
    int_list.append(i)
    
    
  plt.bar(int_list,count)
  plt.xlabel('image_intensities')
  plt.ylabel('intensity-frequencies')
  plt.show()

  
image_histogram('lighthouse.jpg')
