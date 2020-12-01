import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

class histogram_mertics:
    
    
    def __init__(self,pictures,nbins=32):
        """creates a histogram mertics class
        Args:
            pictures: A list of images to caculate historgram mertics on 
            assumes all images have the same shape
            nbins: The number of bins to create the histgram with
        Returns:
            A histogram_mertics object
        """
        self.pictures = tf.constant(pictures,dtype=tf.float64)
        self.histogram = None
        self.cdf = None
        self.nbins = nbins
        self.sess = tf.Session()
    
    
    def image_histogram(self, normalize=False):
        """calculates the histogram mertics
        Args:
            normalize: wheather or not the histgrams are normailized  
        Returns:
            A list of tensors that calculate histograms of each colour channel
        """
        ret_histgrams = []
        for i in range(self.pictures.shape[-1]):
            image = tf.reshape(self.pictures[0,:,:,i], [-1])
                
            maxvalue = tf.math.reduce_max(image)
            minvalue = tf.math.reduce_min(image)
    
            ret = tf.histogram_fixed_width(image, [minvalue,maxvalue],nbins=self.nbins)
            if normalize:
                ret = ret / tf.reduce_sum(ret)
            for x in range(1,self.pictures.shape[0]):
            
                image = tf.reshape(self.pictures[x,:,:,i], [-1])
                
                maxvalue = tf.math.reduce_max(image)
                minvalue = tf.math.reduce_min(image)
                
                hist = tf.histogram_fixed_width(image, [minvalue,maxvalue],nbins=self.nbins)

                if normalize:
                    hist = hist / tf.reduce_sum(hist)
                ret = tf.add(ret,hist)
            ret_histgrams.append(ret)
        self.histogram = ret_histgrams
        return ret_histgrams
    
    def cumulative_distribution(self):
        """calculates the cummulative distribution of the histgrams
        this requires the histgrams to have been calculated 
        Returns:
            A list of tensors that calculate the cummlative distridution of
            histograms of each colour channel
        """
        if not self.histogram:
            self.image_histogram()
        self.cdf = []
        for i in range(self.pictures.shape[-1]):
            img_cdf = tf.math.cumsum(self.histogram[i])
            img_cdf = img_cdf / img_cdf[-1]
            self.cdf.append(img_cdf)
        return self.cdf
    
    def equalize_hist_by_index(self,index):
        """equalizes an inital image, uses the hisograms martics
        Args:
            index: the index of the init picture to equalize
        Returns:
            A tensor that calculates the equlaized image
        """
        if not self.cdf:
            self.cumulative_distribution()
        cdf = self.cdf[0]
        reshaped = tf.reshape(self.pictures[index,:,:,0],[-1])
        interresult = tfp.math.interp_regular_1d_grid(tf.dtypes.cast(reshaped,tf.float64),0,255,cdf)
        output = tf.reshape(reshaped,[self.pictures.shape[1],self.pictures.shape[2],1])
        for i in range(1,self.pictures.shape[-1]):
            cdf = self.cdf[i]
            reshaped = tf.reshape(self.pictures[index,:,:,i],[-1])
            interresult = tfp.math.interp_regular_1d_grid(tf.dtypes.cast(reshaped,tf.float64),0,255,cdf)
            out = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2],1])
            output = tf.concat([output, out],axis=2)
        #note will be in the range 0-1 for constency times it by 255 to get into the right range
        return tf.dtypes.cast(output*255, tf.int8)
    
    def equalize_hist_by_image(self,inimage):
        """equalizes an image passed in, uses the hisograms martics
        Args:
            image: the picture to equalize
        Returns:
            A tensor that calculates the equlaized image
        """
        if not self.cdf:
            self.cumulative_distribution()
        cdf = self.cdf[0]
        if len(inimage.shape) < 3:
            inimage = inimage.reshape(inimage.shape[0],inimage.shape[1],1)
        imshape = inimage.shape
        print(imshape)
        tfimage = tf.constant(inimage)
        reimage = tf.image.resize_image_with_pad(tfimage,self.pictures.shape[1],self.pictures.shape[2])
        reshaped = tf.reshape(reimage[:,:,0],[-1])
        interresult = tfp.math.interp_regular_1d_grid(tf.dtypes.cast(reshaped,tf.float64),0,255,cdf)
        output = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2],1])
        print(output)
        try:
            for i in range(1,imshape[-1]):
                cdf = self.cdf[i]
                reshaped = tf.reshape(reimage[:,:,i],[-1])
                interresult = tfp.math.interp_regular_1d_grid(tf.dtypes.cast(reshaped,tf.float64),0,255,cdf)
                out = tf.reshape(interresult,[self.pictures.shape[1],self.pictures.shape[2], 1])
                output = tf.concat([output, out],axis=2)
        except:
            #note will only get here if init images have less colour channels then inimage
            #we just return whatever channels have been equalized so far
            return output
        #note will be in the range 0-1 for constency times it by 255 to get into the right range
        return tf.dtypes.cast(output*255, tf.int8)
    
    def plot_histogram(self):
        """plots the histgrams
        """
        if self.histogram:
            gram = self.histogram
        else:
            gram = self.image_histogram()
        col = ["r","g","b"]
        for i,x in enumerate(gram):
            ind = self.sess.run(tf.range(self.nbins))
            width = 0.3
            plt.bar(ind - width*(i), self.sess.run(x),width,color = col[i])
        plt.show()
    
    def plot_cdf(self):
        """plots the cdfs
        """
        if self.cdf:
            cdf = self.cdf
        else:
            cdf = self.cumulative_distribution()
        col = ["r","g","b"]
        for i,x in enumerate(cdf):
            ind = self.sess.run(tf.range(self.nbins))
            width = 0.3
            plt.bar(ind - width*(i), self.sess.run(x),width,color = col[i])
        plt.show()