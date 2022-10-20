import train
import dataset
import modules
import predict


def main():
    data = dataset.Dataset()
    module = modules.Modules(data)
    train.handle_training(module)
    predict.handle_prediction(data, module)
    print("ok!")


if __name__ == "__main__":
    main()
