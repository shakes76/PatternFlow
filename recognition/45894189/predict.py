from train import StyleGAN

# Training Variables
EPOCHS = 3
IMAGES_PATH = None
IMAGES_COUNT = 0
WEIGHTS_PATH = None
PLOT_LOSS = False

def main():
    style_gan = StyleGAN(epochs=EPOCHS)
    style_gan.train(images_path=IMAGES_PATH, images_count=IMAGES_COUNT, weights_path=WEIGHTS_PATH, plot_loss=PLOT_LOSS)

if __name__ == "__main__":
    main()