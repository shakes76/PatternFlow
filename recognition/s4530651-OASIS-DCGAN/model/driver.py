import sys
import data
import dcgan

def main(args):
    train, test, real_images = data.process_data(args[1])
    dcgan.train_model(train, int(args[2]), real_images)

if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)