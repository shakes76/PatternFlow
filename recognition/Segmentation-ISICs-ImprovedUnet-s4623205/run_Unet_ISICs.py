import matplotlib.pyplot as plt
from data_utils import get_min_imageshape, load_rgbimages, load_masks


def main():
    image_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"
    mask_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"

    print("Getting minimum image shape...")
    # Image shapes are not consistent, get the minimum image shape. Shape of [283, 340] in this case.
    min_img_shape = get_min_imageshape(mask_path)
    img_height = min_img_shape[0]
    img_width = min_img_shape[1]
    print("\nMin Image Height:", img_height)
    print("Min Image Width:", img_width)

    print("\nLoad and preprocess RGB images...")
    images = load_rgbimages(image_path, img_height, img_width)
    print("\nLoad and preprocess masks...")
    masks = load_masks(mask_path, img_height, img_width)
    print("\nPlotting the first preprocessed RGB image and preprocessed mask...")
    plt.subplot(1, 2, 1)
    plt.imshow(images[0])
    plt.subplot(1, 2, 2)
    plt.imshow(masks[0, :, :, 0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()