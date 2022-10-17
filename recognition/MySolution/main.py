import train
import dataset
import modules


def main():
    data = dataset.Dataset()
    module = modules.Modules(data)
    train.handle_training(module)
    print("ok!")


if __name__ == "__main__":
    main()
