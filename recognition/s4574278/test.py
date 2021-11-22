from PIL import Image
import numpy as np
import utils
from torchsummary import summary

from model import YoloXDecoder

# Path to image to check
to_check = None

# The path to save the result image with bounding box
save_path = ""

# the path to the pre-trained weights .pth file
model_weights_path = None

if __name__ == "__main__":
    validator=YoloXDecoder(model_weights_path, ["lesion"])
    summary(validator.model, input_size=(3, 512, 512))
    image = Image.open(to_check)
    boxes = validator.see(image)

    utils.draw_bbox(np.asarray(image), boxes, save_path)
