from os import sys
from os.path import isfile, join

def predict_model(VQVAETrainer, PixelCNN):
    #TODO predict model
    print("predict with specific model")

def main():
    if (len(sys.argv) == 4 and sys.argv[]):
        print(sys.argv)
        try:
            #TODO load specific model and predict/generate images with loaded model
            print("load DEFAULT model and predict/generate images with loaded model")
        except:
            #TODO  Load default model as regular model was able to be predicted with
            print('loading defualt model')
    elif (len(sys.argv == 1)):
        #Check if default model has been trained & saved:
        #TODO load DEFAULT model and predict/generate images with loaded model
        print("load DEFAULT model and predict/generate images with loaded model")
    else:
        print("$ python3 predict.py [-m <PathToPreTrainedModel>]")

if __name__ == "__main__":
    main()