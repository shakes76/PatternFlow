from PIL import Image
import numpy as np
import torch
import utils
from torchsummary import summary

from model import ModelSize, YoloxModel

to_check = ""
save_path = ""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloxModel(num_classes=1, model_size=ModelSize.S).to(device)
    model.load_state()
    summary(model, input_size=(3, 512, 512))
    image = Image.open(to_check)
    box = model.see(image)

    utils.draw_bbox(np.asarray(image), [box], save_path)
