import GANutils
import modules

def generate_images(model_folder: str = "model/", num_images = 1, save_folder = None) -> None:
    #TODO docstring
    #Constants relating to the given training set (again minor duplication due to ensuring everything is wrapped in a class/function)
    LATENT_DIM = 512
    NOISE_START = 4
    NOISE_END = 32

    model = modules.StyleGAN(existing_model_folder= model_folder)
    for i in range(num_images):
        #Name files to be generated inside create_image if applicable
        image_name = save_folder
        if save_folder is not None:
            image_name += str(i)

        GANutils.create_image(GANutils.denormalise(model(GANutils.random_generator_inputs(num_images, LATENT_DIM, NOISE_START, NOISE_END))[0],model.get_mean()), image_name).show()
