import tensorflow as tf
import numpy as np
def Sobel(image):
	"""
	Edge Detection using Sobel transform.

	Parameters:
	-----------

	image : 2-D array (Image to process)

	Returns:
	--------

	resultant : 2-D array (The edge map of the image)

	Notes:
	------
	Square root of the squares of the horizontal and vertical edges are taken
	to get a magnitude i.e. somewhat insensitive to direction.


	"""

	tf.reset_default_graph()

	#Horizontal Sobel kernel
	#Sobel_h = np.array([3,3])
	Sobel_h = [ [1,2,1], [0,0,0], [-1,-2,-1] ]
	#Vertical Sobel kernel
	#Sobel_v = np.array([3,3])
	Sobel_v = [ [1,0,-1], [2,0,-2], [1,0,-1] ]

	input_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))

	with tf.name_scope('Convolution'):
		kernel_h = tf.constant(Sobel_h, dtype=tf.float32, shape=(3,3,1,1))
		kernel_v = tf.constant(Sobel_v, dtype=tf.float32, shape=(3,3,1,1))
		output_h = tf.nn.conv2d(input=input_placeholder, filter=kernel_h, strides=[1,1,1,1], padding="SAME")
		output_v = tf.nn.conv2d(input=input_placeholder, filter=kernel_v, strides=[1,1,1,1], padding="SAME")  
    
	with tf.Session() as sess:
		h_edges = sess.run(output_h, feed_dict={input_placeholder: image[np.newaxis,:,:,np.newaxis]})
		v_edges = sess.run(output_v, feed_dict={input_placeholder: image[np.newaxis,:,:,np.newaxis]})

	#plt.imshow(h_edges[0,:,:,0], cmap="hot")
	#plt.show()
	#plt.imshow(v_edges[0,:,:,0], cmap="hot")
	#plt.show()


	resultant = ((h_edges**2) + (v_edges**2))**0.5
	#plt.imshow(resultant[0,:,:,0], cmap="hot")
	return resultant