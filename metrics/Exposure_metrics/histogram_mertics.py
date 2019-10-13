import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import itertools
import numpy as np
from math import factorial
import tensorflow_probability as tfp
from functools import reduce

class histogram_mertics:
    
    
    def __init__(self,pictures,nbins=256):
        self.pictures = pictures.astype(np.float64)
        self.histogram = None
        self.cdf = None
        self.nbins = nbins
    
    
    def image_histogram(self, normalize=False):
        ret_histgrams = []
        for i in range(self.pictures.shape[-1]):
            print(i)
            image = tf.reshape(self.pictures[0,:,:,:], [-1])
                
            maxvalue = tf.math.reduce_max(image)
            minvalue = tf.math.reduce_min(image)
    
            ret = tf.histogram_fixed_width(image, [minvalue,maxvalue],nbins=self.nbins)
            if normalize:
                ret = ret / tf.reduce_sum(ret)
            for x in self.pictures[1:,:,:,:]:
            
                image = tf.reshape(x, [-1])
                
                maxvalue = tf.math.reduce_max(image)
                minvalue = tf.math.reduce_min(image)
                
                hist = tf.histogram_fixed_width(image, [minvalue,maxvalue],nbins=self.nbins)
            
                if normalize:
                    hist = hist / tf.reduce_sum(hist)
                ret = tf.add(ret,hist)
            ret_histgrams.append(ret)
        self.histogram = ret_histgrams
        return ret_histgrams
    
    def equalize_hist_by_index(self,index):
        if not self.cdf:
            self.cumulative_distribution()
        cdf = self.cdf[0]
        reshaped = tf.reshape(self.pictures[index,:,:,0],[-1])
        interresult = tfp.math.interp_regular_1d_grid(reshaped,0,255,cdf)
        output = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2],1])
        for i in range(1,self.pictures.shape[-1]):
            cdf = self.cdf[i]
            reshaped = tf.reshape(self.pictures[index,:,:,i],[-1])
            interresult = tfp.math.interp_regular_1d_grid(reshaped,0,255,cdf)
            out = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2],1])
            output = tf.concat([output, out],axis=2)
        return output
    
    def equalize_hist_by_image(self,image):
        if not self.cdf:
            self.cumulative_distribution()
        cdf = self.cdf[0]
        reshaped = tf.reshape(image,[-1])
        interresult = tfp.math.interp_regular_1d_grid(reshaped,0,255,cdf)
        output = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2],1])
        for i in range(1,image.shape[-1]):
            cdf = self.cdf[i]
            reshaped = tf.reshape(image[:,:,i],[-1])
            interresult = tfp.math.interp_regular_1d_grid(reshaped,0,255,cdf)
            out = tf.reshape(interresult,[image.shape[0],image.shape[1],1])
            output = tf.concat([output, out],axis=2)
        return output

    def cumulative_distribution(self):
        if not self.histogram:
            self.image_histogram()
        self.cdf = []
        for i in range(self.pictures.shape[-1]):
            img_cdf = tf.math.cumsum(self.histogram[i])
            img_cdf = img_cdf / img_cdf[-1]
            self.cdf.append(img_cdf)
        return self.cdf
    
(x_train, y_train), (x_eval, y_eval) = cifar10.load_data()
x_eval = x_eval[:11]
print(x_eval.shape)
x = histogram_mertics(x_eval)
print(x)
x.image_histogram()
x.cumulative_distribution()
print(x.equalize_hist_by_index(0))