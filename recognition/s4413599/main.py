from DSC_FUNCTION import DSC,DSC_LOSS
from image_load import get_train_test_data
from improved_UNET import improved_UNET
import matplotlib.pyplot as plt
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test_data()
    model = improved_UNET()
    print(model.summary())
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [DSC, 'accuracy'])
    history = model.fit(x=X_train, y=y_train, epochs=20, batch_size=28, validation_split = 0.1)

    # Set the threshold. If value larger than 0.5 set to 1 otherwise set to 0
    pre_test = model.predict(X_test, verbose=1)
    pre_test[pre_test > 0.5] = 1
    pre_test[pre_test <= 0.5] = 0

    # Plot sample images (Actual test image, actual segmentation image, predicted segmentaion image)
    f, axes = plt.subplots(3, 3)
    for i in range(3):
        axes[i, 0].imshow(X_test[i], cmap='gray')
        axes[i, 1].imshow(y_test[i], cmap='gray')
        axes[i, 2].imshow(pre_test[i], cmap='gray')
    # Calculate the Dice similarity coefficient