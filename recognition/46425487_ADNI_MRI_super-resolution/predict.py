import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# import the model and test data set created by train.py
model = keras.models.load_model('./trained_model')
test_ds = tf.data.Dataset.load('./test_data')

def svplot(img, title, index):

    #Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img[::-1], origin="lower", cmap="gray", vmin=0, vmax=1)

    plt.title(title)
    # zoom by a factor of two and plot it in the lower left of image
    axins = zoomed_inset_axes(ax, 2, loc=3)
    axins.imshow(img[::-1], origin="lower", cmap="gray", vmin=0, vmax=1)

    # Specify the zoom area
    x1, x2, y1, y2 = 100, 160, 100, 160
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # draw lines from zoom area to zoomed in plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="blue")
    plt.savefig("./results/" + str(index) + "-" + title + ".png")


# reconstruct low res images from test dataset using the model and compare them to the originals
total_resize_psnr = 0.0
total_predic_psnr = 0.0

for batch in test_ds.take(1):
    # get the low_res and high_res images from test dataset
    for i, low_res in enumerate(batch[0]):
        high_res = batch[1][i]

        # create a reconstructed image from the low_res image using the model  
        inp = np.expand_dims(low_res, axis=0)
        predicted = model.predict(inp)
        predicted = predicted[0]
        
        # resize low res image to match the others
        low_res = keras.preprocessing.image.array_to_img(low_res)
        low_res = low_res.resize((256, 256))
        low_res = keras.preprocessing.image.img_to_array(low_res)
        low_res = low_res / 255.0

        resize_psnr = tf.image.psnr(low_res, high_res, max_val=1)
        predic_psnr = tf.image.psnr(predicted, high_res, max_val=1)

        total_resize_psnr += resize_psnr
        total_predic_psnr += predic_psnr


        print(f"PSNR of low res and high res for image {i} is {resize_psnr:.4f}")
        print(f"PSNR of predicted and high res for image {i} is {predic_psnr:.4f}")

        svplot(low_res, "low-res", i)
        svplot(high_res, "high-res", i)
        svplot(predicted, "predicted", i)

print(f"Average PSNR of low res and high res for images is {total_resize_psnr/32:.4f}")
print(f"Average PSNR of predicted and high res for images is {total_predic_psnr/32:.4f}")