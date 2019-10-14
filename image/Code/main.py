from histogram import *

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
with tf.Session() as sess:
  ax[0].imshow(tf.reshape(image,[512,512]).eval(), interpolation='nearest', cmap=plt.cm.gray)
  ax[0].axis('off')

  ax[1].plot(hist_centers, hist, lw=2)
  ax[1].set_title('Histogram of grey values')
