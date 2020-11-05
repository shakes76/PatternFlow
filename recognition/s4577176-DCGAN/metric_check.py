from libraries import *
from helper_functions import *

dir_data = "C:\\Users\\61484\\Desktop\\comp3710-demo3\\particles.js-master\\keras_png_slices_data\\keras_png_slices_train"
Ntrain        = len(os.listdir(dir_data))
nm_imgs       = np.sort(os.listdir(dir_data))
img_shape = (64,64,3)

img_id = np.random.choice(nm_imgs)
image = load_img(dir_data + "/" + img_id,target_size=img_shape[:2])
image = img_to_array(image)/255.0


generator_model = load_model("C:\\Users\\61484\\comp3710_final\\PatternFlow\\recognition\\s4577176-DCGAN\\saved_models\\generator" + "-" + str(img_shape[0]) + ".h5")
noise = get_noise(1)
rnd_img = generator_model.predict(noise)[0]


structural_similarity = tf.image.ssim(tf.constant(image),tf.constant(rnd_img),max_val=1.0).numpy()

print(structural_similarity)