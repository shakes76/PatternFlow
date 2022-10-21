# -*- coding: utf-8 -*-
"""
Created on Oct 30, 2020

@author: s4542006, Md Abdul Bari
"""

import pandas as pd
from train_test_eval import model_train_test, plot_sample_image, get_loss_plot


def main():
    """Provide model performance and pertinent plots"""
    # store model's predictive performance resutls and training stats
    res_dict, hist_dict = model_train_test(n_epochs=50, batch_size=32)
    class_dsc = res_dict["class_dsc"]
    dsc_back = class_dsc[0]
    dsc_fore = class_dsc[1]
    overall = res_dict["overall_dsc"]
    print("\nThanks for your inputs and patient waiting!")
    print("\n\nDice Similarity Coefficient (DSC) on test set:\n")
    dict_dsc = {"CATEGORY": ["Background", "Lesion", "Overall"], 
                "DSC": [dsc_back, dsc_fore, overall]}
    # create and output data frame showing class-wise and overall DSC     
    df_dsc = pd.DataFrame(dict_dsc)
    print(df_dsc.to_string(index=False))
    # save results in .csv format
    df_dsc.to_csv("45420065_test_dsc.csv", index=False)
    # store and plot model training and validation loss
    history = hist_dict["history"]
    get_loss_plot(hist=history)
    # sample plots of input, ground truth and predicted images from testset
    plot_sample_image(res_dict["input_test"], res_dict["gt_test"], 
                      res_dict["pred_test"], ix=45)
    plot_sample_image(res_dict["input_test"], res_dict["gt_test"], 
                      res_dict["pred_test"], ix=218)
    
    
if __name__ == "__main__":
    main()