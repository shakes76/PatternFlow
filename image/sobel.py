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
	#Horizontal Sobel kernel

	#Horizontal Sobel kernel
	Sobel_h = np.array([1,2,1], [0,0,0], [-1,-2,-1])
	#Vertical Sobel kernel
	Sobel_v = np.array([1,0,-1], [2,0,-2], [1,0,-1])

	input_placeholder = tf.input_placeholder(dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))

	kernel_h = tf.constant(Sobel_h, dtype=tf.float32, shape=(3,3,1,1))
	kernel_v = tf.constant(Sobel_v, dtype=tf.float32, shape=(3,3,1,1))
	output_h = tf.nn.conv2d(input=input_placeholder, filter=kernel_h, strides=[1,1,1,1], padding="SAME")
	output_v = tf.nn.conv2d(input=input_placeholder, filter=kernel_v, strides=[1,1,1,1], padding="SAME")

	with tf.Session() as sess:
		h_edges = sess.run(output_h, feed_dict={input_placeholder: image[np.newaxis,:,:,np.newaxis]})
		v_edges = sess.run(output_v, feed_dict={input_placeholder: image[np.newaxis,:,:,np.newaxis]})


	return h_edges, v_edges