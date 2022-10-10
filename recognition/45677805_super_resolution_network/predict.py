from modules import *
import tensorflow as tf
from dataset import *
import matplotlib.pyplot as plt

def loadModel_from_path(has_model):
    model = get_model()
    if(has_model):
        model.load_weights("G:/Australia/academic/UQ/2022 s2/comp 3710/A3/s4567780 Problem 5  super-resolution network/PatternFlow/recognition/45677805_super_resolution_network/model/modeH5_148epochs.h5")
    return model


# compare the result of original,resized and model predicted
def predict_comparion(test_ds, model):
    """ 改进版 解决了图片不是同一张问题"""
    print("original----------")
    test_one_img_data = get_img(test_ds)
    # print("test_one_img_data: ", test_one_img_data.shape) # test_one_img_data:  (128, 128, 3)
    test_img_one = array_to_img(test_one_img_data) 
    display(test_img_one)

    # plt.imshow(test_img_one)
    # plt.show()

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


    # plt method 
    fig = plt.figure(figsize=(128,128))

    # fig.tight_layout(h_pad=15)
    # ax1 = fig.add_subplot(311)
    # # ax1.axis('off')
    # ax2 = fig.add_subplot(312)
    # # ax2.axis('off')
    # ax3 = fig.add_subplot(313)
    # # ax3.axis('off')
    # ax1.set_title("original img")
    # ax2.set_title("resized 128->32->128 by bicubic method ")
    # ax3.set_title("restore by srcnn model")
    
    # ax1.imshow(test_img_one)
    # ax2.imshow(resized_test_one_img)
    # ax3.imshow(predict_img_one)

    img_list = [test_img_one, resized_test_one_img, predict_img_one]

    for i in range(1,4):
        fig.add_subplot(3, 1, i)
        img = img_list[i-1]
        
        plt.imshow(img)
    # plt.savefig("PatternFlow/recognition/45677805_super_resolution_network/predict_result/epoch_148.png")    
    plt.show()
    
def predict():
    model = loadModel_from_path(True)
    test_data = get_test()
    predict_comparion(test_data, model)

predict()
# fig = plt.figure(figsize=(128,128))
# fig.tight_layout(h_pad=15)
# ax1 = fig.add_subplot(311)
# # ax1.axis('off')
# ax2 = fig.add_subplot(312)
# # ax2.axis('off')
# ax3 = fig.add_subplot(313)
# # ax3.axis('off')
# ax1.set_title("original img")
# ax2.set_title("resized 128->32->128 by bicubic method ")
# ax3.set_title("restore by srcnn model")

# plt.show()