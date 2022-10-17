# We use SSIM as our loss function becase it performs better than MSE
# and is only dependant on luminance (which is all we care about) given
# that we are working with greyscale images