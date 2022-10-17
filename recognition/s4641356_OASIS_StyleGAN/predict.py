import GANutils
import modules
import os

def generate_images(
            model_folder: str = "model/", 
            num_images: int  = 1, 
            save_folder: str = None
            ) -> None:
    """
    Uses the trained styleGAN to generate images of OASIS dataset brains.
    This function generates random inputs to seed each image, but if you desire 
    to use your own specified input, you can call model instance directly on 
    your inputs. Images are shown in their own window and also saved to disk if 
    requested.

    Args:
        model_folder (str, optional): Folder containing the styleGAN model to 
                generate images from. Defaults to "model/".
        num_images (int, optional): Number of images to produce. Defaults to 1.
        save_folder (str, optional): Folder in which to save generated images. 
                If the folder does not exist it is created. If None images are 
                not saved to disk. Defaults to None.
    """
    #Constants relating to the given training set
    #minor duplication tolerated to keep everything wrapped in a class/function
    LATENT_DIM = 512
    NOISE_START = 4
    NOISE_END = 32

    #styleGAN model we call on random input to generate image data
    model = modules.StyleGAN(existing_model_folder= model_folder)

    if (save_folder is not None) and (not os.path.exists(save_folder)):
        os.makedirs(save_folder)

    for i in range(num_images):
        #Name files to be generated inside create_image if applicable
        image_name = save_folder
        if save_folder is not None:
            image_name += str(i)

        #Generate image. Functionality explained in GANutils.create_image 
        GANutils.create_image(
                GANutils.denormalise(
                    model(
                        GANutils.random_generator_inputs(
                            num_images, 
                            LATENT_DIM, 
                            NOISE_START, 
                            NOISE_END
                            )
                        )[0],
                    model.get_mean()
                    ), 
                image_name
                ).show()
