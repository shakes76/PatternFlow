# Chosen algorithm: L0 norm smoothing of images
#
# author: naziah siddique (44315203)

from PIL import Image
import tensorflow as tf 
import numpy as np
import argparse 

# Require tensorflow version >=2.0 to run
print(tf.__version__)

from l0_norm_smoothing import l0_calc


def main(imdir, outdir, _lambda, kappa, beta_max):

    # load image into array 
    tf_img = tf.keras.preprocessing.image.load_img(imdir)
    img_arr = np.array(tf_img)

    # pass image and calculate and output gradient smoothing 
    out_img = l0_calc(img_arr, _lambda, kappa, beta_max)
    
    # save image from output array 
    im = Image.fromarray(out_img.astype(np.uint8))
    im.save(outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="L0 Gradient Smoothing")
    parser.add_argument("-d", "--inputimgdir", dest="imgdir",
                        help="Directory path for input image",
                        metavar="FILE", default='example/dahlia.png')
    parser.add_argument("-o", "--outdir", dest="outdir",
                        help="Directory path for output image",
                        metavar="FILE", default="example/dahlia_out.png")
    parser.add_argument("-l", "--lamdaval", dest="lamdaval",
                        help="lambda parameter",
                        metavar="FLOAT", default=2e-2)
    parser.add_argument("-k", "--kappa", dest="kappa",
                        help="kappa parameter",
                        metavar="FLOAT", default=2.0)
    parser.add_argument("-b", "--beta_max", dest="beta_max",
                        help="beta max parameter",
                        metavar="FLOAT", default=1e5)


    args = parser.parse_args()

    main(args.imgdir, 
        args.outdir, 
        float(args.lamdaval), 
        float(args.kappa), 
        float(args.beta_max))



