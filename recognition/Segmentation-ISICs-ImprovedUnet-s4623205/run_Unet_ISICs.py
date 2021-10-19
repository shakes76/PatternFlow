import matplotlib.pyplot as plt
import cv2

def main():
    img_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1-2_Training_Input_x2\ISIC_0000000.jpg"

    print("Reading single RGB image...")
    # OpenCV reads images as BGR, convert it to RGB when reading
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    print("Plotting single RGB image...")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()