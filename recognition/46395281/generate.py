import sys
import torch
from model import Gnet
import numpy as np
import matplotlib.pyplot as plt
def main(arglist):
    """
    :param arglist:
        arglist[0]=model_path, specifying the directory of the trained model
        arglist[1]=sample_path, specifying the directory to save the results including samples and one .gif
    :return:
    """
    model_path = arglist[0]
    sample_path = arglist[1]
    #The below two lines are example usage on Google Colab
    #model_path = 'drive/MyDrive/training_results_style/G.pth'
    #sample_path = 'drive/MyDrive/training_results_style/'

    #set the device
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    #create and load the model
    G = Gnet().to(device)
    params = torch.load(model_path, map_location=device)
    G.load_state_dict(params["state_dict"])
    #set to eval mode, for generating images only
    G.eval()
    noise = torch.randn(32, 512, device=device)
    generated = (G(noise, alpha=0, steps=6) * 0.5) + 0.5 #transfer the image to range[0,1]
    for i, img in enumerate(generated):
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0)) #transfer the format
        img = img * 255#transfer the image to range[0,255]
        plt.imsave(sample_path + 'generated_' + str(i) + '.png', img.reshape(256, 256), cmap='Greys_r')#save the image
    #back to training mode, not neccessary
    G.train()

if __name__ == "__main__":
    main(sys.argv[1:])