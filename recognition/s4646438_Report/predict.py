"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from dataset import input_downsample
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Visualiser(object):
    '''Class which allows the visualisation of the output of the model'''
    def produce_comparison_plot(self, input_path):
      '''
      Produces a set of images. Original image, a scaled up version using bicubic resizing and the output from my model
      '''
      #load in the test image, pad and resize it to pass into the model
      original_image = img_to_array(load_img(input_path))
      original_image = tf.image.pad_to_bounding_box(original_image, 0, 0, target_image_width, target_image_width)
      test_image = np.expand_dims(original_image, axis=0)
      original_image = array_to_img(original_image)
      test_image = input_downsample(test_image, target_image_width, upsample_factor)
      scaled = np.squeeze(test_image, axis=0)
      scaled = array_to_img(tf.image.resize(scaled, (target_image_width, target_image_width), method='bicubic'))
      
      display(original_image)
      display(scaled)
      display(model.predict(test_image))
