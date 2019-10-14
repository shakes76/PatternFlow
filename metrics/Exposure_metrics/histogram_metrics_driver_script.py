from histogram_metrics import histogram_mertics
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10, mnist 
import matplotlib.pyplot as plt

if __name__== "__main__" :
    #load some sample data and take a subset so computation isn't too long
    (x_train, y_train), (x_eval, y_eval) = cifar10.load_data()
    x_eval = x_eval[:20]
    x = histogram_mertics(x_eval)
    
    #plot the images histograms and cdf, the calculations will happening in the right order tahnks to the object
    x.plot_histogram()
    x.plot_cdf()
    
    #equalize an image we passed in in the init
    with tf.Session() as sess:
        plt.imshow(sess.run(x.pictures[0,:,:,:]).astype(int))
        plt.show()
    
    with tf.Session() as sess:
        equalized_image = x.equalize_hist_by_index(0)
        plt.imshow(sess.run(equalized_image))
        plt.show()
        
    #equalize an image we didn't pass in at init
        
    (x_train, y_train), (x_eval, y_eval) = mnist.load_data()
    plt.imshow(x_train[0])
    plt.show()
    
    with tf.Session() as sess:
        equalized_image = x.equalize_hist_by_image(x_train[0])
        plt.imshow(sess.run(equalized_image).reshape(32,32))
        plt.show()
