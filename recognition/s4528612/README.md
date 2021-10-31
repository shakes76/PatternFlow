# Perceiver Model Implemtation for Classification of the OAI AKOA knee data laterality

# Perceiver

The Percevier model is a model which builds on transformers while relaxing the assumptions on the relationships between inputs while still being able to scale to a large quantity of inputs. Released by Google Deepmind in June 2021, the model allows for flexibility with the input types and so is used here to classify the laterality of human knees provided by the OAI AKOA Knee Dataset. 

# Data

The OAI AKOA consists of 18680 images from 427 patients with 7760 left knee and 10920 right knee images. The data was processed by reshaping to a size of (73,64) which proved to be successful for other prac students as it saved time while still maintaing accuracy. Furthermore, the images were converted to greysacle and normalized by diviging by 255.

# Architecture 

Cross-Attention 
 ...

Fourier Encoder
To ensure that the order of the pixels is maintained, a Fourier Encoding was applied to the inputs. The Fourier Layer included code from Rishit Dagli, linked here: https://github.com/Rishit-dagli/Perceiver. 


# Results
Test result of 0.988 on 4% of the data.