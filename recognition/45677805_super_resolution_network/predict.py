from modules import *
import tensorflow as tf
from dataset import *
import matplotlib.pyplot as plt

"""load the trained model from the path"""
def loadModel_from_path(has_model):
    model = get_model()
    if(has_model):
        model.load_weights("model\modeH5_265.h5")
    return model


"""output the model result image and comparing with original, reconstructed by tensorflow bicubic interpolation"""
def predict_comparion(test_ds, model):

    """presenting in .ipynb format"""
    print("original----------")
    test_one_img_data = get_img(test_ds)
    test_img_one = array_to_img(test_one_img_data) 
    display(test_img_one)

    print("resized----------")
    resized_img_array = resize_img(test_one_img_data, 32)
    resized_test_one_img = array_to_img(resized_img_array) 
    display(resized_test_one_img)
    print(resized_img_array.shape) #(128, 128, 3)
    resized_img_array = tf.expand_dims(resized_img_array, axis = 0) # for feed into model prediction
    print(resized_img_array.shape) # (1, 128, 128, 3)

    print("----------predict----------")
    predict_result_one_data = model.predict(resized_img_array)
    #print(predict_result_one_data.shape) # (1, 128, 128, 3)
    predict_img_one = array_to_img(predict_result_one_data[0])
    display(predict_img_one)


    """presenting in .py file""" 
    fig = plt.figure(figsize=(128,128))

    small_img_data = tf.image.resize(test_one_img_data, [32, 32], method='bicubic')
    small_img = array_to_img(small_img_data) 
    
    img_list = [test_img_one, small_img, resized_test_one_img, predict_img_one]

    # plot in one frame
    for i in range(1,5):
        fig.add_subplot(5,1, i)
        img = img_list[i-1]
        
        plt.imshow(img)
    plt.show()
    """ for saving the picture"""
    # plt.savefig("PatternFlow/recognition/45677805_super_resolution_network/predict_result/epoch_265.png")    
    # plt.savefig("PatternFlow/recognition/45677805_super_resolution_network/predict_result/result_comparison.png")

    """for saving the resized small image"""
    # fig1 = plt.figure(figsize=(128,128))
    # ax1 = fig1.add_subplot(111)
    # ax1.imshow(small_img)  
    # plt.savefig("PatternFlow/recognition/45677805_super_resolution_network/predict_result/small.png")
    # plt.show()

    
def predict():
    model = loadModel_from_path(True)
    test_data = get_test()
    predict_comparion(test_data, model)

predict()
