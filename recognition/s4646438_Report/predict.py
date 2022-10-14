"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

    def plot_results(self, img, prefix, title):
        """
        Plot the result with zoom-in area.
        FUNCTION TAKEN FROM REFERENCE 1
        """
        img_array = img_to_array(img)
        img_array = img_array.astype("float32") / 255.0

        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title(title)
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 300, 100, 200
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
        plt.savefig(str(prefix) + "-" + title + ".png")
        plt.show()
