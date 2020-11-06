import sys
from load_data import *
from model import *
import numpy as np
from PIL import Image

if __name__ == "__main__":
    #gets inputed path
    path = sys.argv[1]
    #produces directory for generated images
    save_dir = os.path.join(path,'gen_images_01')
    #changes path for where data should be located
    path = os.path.join(path, 'keras_png_slices_data')
    path = os.path.join(path, 'keras_png_slices_seg_train')
    #gets array of paths 
    image_paths = load_paths(path)
    image_data = tf.data.Dataset.from_tensor_slices(image_paths)

    #makes save director
    os.mkdir(save_dir) 
    #generator for images
    image_data = image_data.map(load_images)
    image_batch = iter(image_data.batch(32))

    disc = build_discriminator((256, 256, 3))
    gen = build_generator(100)
    
    opt1 = tf.keras.optimizers.Adam(0.0001)
    opt2 = tf.keras.optimizers.Adam(0.0001)

    step = 0
    while True:
        print("epoch: ", step)
        gen_image = None
        data = None
        while True:
            try:            
                results = next(image_batch)
                gen_image = train_step(results, disc, gen, opt1, opt2)
                
            except StopIteration:
                step += 1
                
                img = Image.fromarray((gen_image[0].numpy()*255).astype('uint8'), mode='RGB')
            
                img.save(os.path.join(save_dir, "generated_img" + str(step) + ".png"))

                image_batch = iter(image_data.batch(32))

                break
    
