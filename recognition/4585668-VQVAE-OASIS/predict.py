import tensorflow as tf

from numpy.random		import choice
from tensorflow.image	import convert_image_dtype, ssim
from matplotlib.pyplot	import suptitle, imshow, subplot, axis, show, cm, title
from keras.models		import load_model

from dataset			import get_ttv, normalise
from modules			import VectorQuantiser
from train				import MODEL_PATH

RECONS_TO_VIEW	= 5
COLS			= 2
MAX_VAL			= 1.0
CENTRE			= 0.5
GREY			= cm.gray

def compare(images, recons):
	ssims = 0
	for i, pair in enumerate(zip(images, recons)):
		o, r = pair
		orig = convert_image_dtype(o, tf.float32)
		recon = convert_image_dtype(r, tf.float32)
		sim = ssim(orig, recon, max_val = MAX_VAL)
		ssims += sim
		subplot(RECONS_TO_VIEW, COLS, COLS * i + 1)
		imshow(o + CENTRE, cmap = GREY)
		title("Test Input")
		axis("off")
		subplot(RECONS_TO_VIEW, COLS, COLS * (i + 1))
		imshow(r + CENTRE, cmap = GREY)
		title("Test Reconstruction")
		axis("off")
		suptitle("SSIM: %.2f" %sim)
	show()

	return ssims

def main():
	train, te, validate = get_ttv()
	test = normalise(te)
	vqvae = load_model(MODEL_PATH, custom_objects = {"VectorQuantiser": VectorQuantiser})
	image_inds = choice(len(test), RECONS_TO_VIEW)
	images = test[image_inds]
	recons = vqvae.predict(images)
	ssim = compare(images, recons)
	avg_ssim = ssim / RECONS_TO_VIEW
	print(f"Average SSIM: {avg_ssim}")

if __name__ == "__main__":
	main()
