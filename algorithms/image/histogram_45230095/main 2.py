from histogram import *

# Use the built-in data image "camera" for the demonstration example
noisy_image = img_as_ubyte(data.camera()) 
# Cast the array type to int32 to avoid type error
tf.dtypes.cast(noisy_image, tf.int32)
# This is where the function is used
hist, hist_centers = histogram(noisy_image)

# Plotting the origin picture and the histogram
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(noisy_image, interpolation='nearest', cmap=plt.cm.gray)
ax[0].axis('off')

ax[1].plot(hist_centers, hist, lw=2)
ax[1].set_title('Histogram of grey values')

plt.tight_layout()
