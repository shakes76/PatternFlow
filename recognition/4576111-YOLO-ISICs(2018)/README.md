# Lesion Detection with YoloV1
This is a python based package that utilizes a custom YoloV1 model to detect skin lesions.
## Description
#### Background 
Australia has one of the highest rates of deaths by skin cancer in the world[1]. Detecting problematic skin lesions early is one of the best preventative methods for stopping the progression life threating cancer. The most common method in peforming this detecting is seeing a dermatologist[2], unfortunely this depends on the ability of dermatologist and the results can be varied[3]. 

#### Solution
This module gives a partial solution to this problem. Using the help of Convolutional Neural Networks, this module provides the ability for skin lesions to be detected using image detection, this can aid a dermatologist in their search for cancerous skin lesions. Further work will go into classifying these lesions. 

### Model Architecture:
This module utilizes a slight variation on the YoloV1 Architecture. 
![test](https://imgur.com/39WImDQ)
At each CNN layer, Batch Normalisation has been introduced as a way of speeding up the training process. 
### Dataset:

## References:
1. Sinclair, C. and Foley, P. (2009). Skin cancer prevention in Australia. British Journal of Dermatology, 161, pp.116–123.
2. Aitken, J.F., Janda, M., Lowe, J.B., Elwood, M., Ring, I.T., Youl, P.H. and Firman, D.W. (2004). Prevalence of Whole-Body Skin Self-Examination in a Population at High Risk for Skin Cancer (Australia). Cancer Causes & Control, 15(5), pp.453–463.
3. Jerant, A.F., Johnson, J.T., Sheridan, C.D. and Caffrey, T.J. (2000). Early Detection and Treatment of Skin Cancer. American Family Physician, [online] 62(2), pp.357–368. Available at: https://www.aafp.org/afp/2000/0715/p357.html?searchwebme [Accessed 9 Oct. 2020]