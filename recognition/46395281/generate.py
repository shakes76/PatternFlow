import sys
import torch
from model import Gnet
import numpy as np
import matplotlib.pyplot as plt
def main(arglist):
    model_path = arglist[0]
    sample_path = arglist[1]
    #model_path = 'drive/MyDrive/training_results_style/G.pth'
    #sample_path = 'drive/MyDrive/training_results_style/'

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    G = Gnet().to(device)
    params = torch.load(model_path, map_location=device)
    G.load_state_dict(params["state_dict"])
    G.eval()
    noise = torch.randn(32, 512, device=device)
    generated = (G(noise, alpha=0, steps=6) * 0.5) + 0.5
    for i, img in enumerate(generated):
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
        img = img * 255
        plt.imsave(sample_path + 'generated_' + str(i) + '.png', img.reshape(256, 256), cmap='Greys_r')
    G.train()
if __name__ == "__main__":
    main(sys.argv[1:])