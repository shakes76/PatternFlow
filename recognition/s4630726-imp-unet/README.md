# IMPROVED UNET 
#### _Welcome to the improved UNET!_

The improved unet utilises several different changes from the one in demo 2. For example, the verboseness is reduced by the means of simplifying repeated patterns of convolutions and other keras operations into individual modules that are easy to implement when one of the modules pop up. Another change is the structure towards the end where the individal localization outputs are summed together before going through the activation layer. The use of this unet model is for the goal of segementing moles from the ISIC melenoma data set with an average DSC score of at least 0.8 for both classes (moles and skin).

#### HOW TO USE

To use this version of improved unet. First make sure you have the ISIC data set found at https://cloudstor.aarnet.edu.au/sender/?s=download&token=723595dd-15b0-4d1e-87b8-237a7fe282ff (delete the txt files at the beginning and end).

You will also need to make sure all the dependencies are installed (see the dependencies list below).

Then open up main.py in your text editor and change the "root" variable to where ever the ISIC data is stored. In that directory there should be two folders: 
* "ISIC2018_Task1-2_Training_Input_x2"
* "ISIC2018_Task1_Training_GroundTruth_x2"

Then using your console navigate to where the directory s4630726-imp-unet is located.

Type in the console "python main.py" and then the script will run all the way through untill the end where the DSC scores are calculated for each class. 

#### DEPENDENCIES:
* Anaconda3 (or the substituents used in the code that are Anaconda3)
* tensorflow
* os for directory operations
* cv2 for image reading
* scikit for splitting the data into training and test sets
* scikit image for image manipulation (resizing etc)

#### RESULTS: 
The model definitely works and segements moles, some very accurately and some not so good. This is unfortuneately reflected in the DSC score where the best I was alble to get was ~0.76. Not able to achieve the goal of 0.8 DSC for each class, here's an example case.

##### *The input image*
![Figure_1](https://user-images.githubusercontent.com/92525563/139614788-cbede4e2-4d6b-4051-9959-03699af1cb20.png)

##### *The ground truth mask*
![Figure_2](https://user-images.githubusercontent.com/92525563/139614791-e472be8f-06aa-4b2f-8e60-4aaf52a2af8a.png)

##### *Predicted mask*
![Figure_3](https://user-images.githubusercontent.com/92525563/139614786-0af5fa3c-8a04-470d-9fb8-4e6932dbac37.png)


#### CHANGES MADE TO ACHIEVE RESULTS / HYPERPARAMETERS:
* the biggest notable change happened afer changing from a grayscale implementation to a colour one (GBR). The DSC scores went from ~0.71-0.72 to ~0.75-0.76
* activation function: sigmoid
* loss function: DSC
* learning rate (stastic): 1e-5
* threshold for predictions: 0.91
* epochs: 100 (for best results)
* batch size: 20
* optimizer: Adam

#### PROCEDURE:
1. The data is first loaded into each array iteratively where it is resized to a 4:3 aspect ratio at 256 x 192 resolution, that being the most accurate representation of the majority of the images in the dataset. This is because all the images are different resolutions and aspect ratios. 

1. Then the arrays are split into testing and training sets with a test size of 0.2. This is the optimal test split ratio.

1. Next the 4 arrays are turned into tensors with datatype float32. The X tensors are normalized by /255, and the Y tesnors get an extra dimensione added so they are the correct format.

1. This data is then fed into the unet, during training there were issues where the loss function would sheldom change and thus would not segment properly. This was fixed by decreasing the learning rate to 1e-5. A split of 0.1 is used for the validation split when fitting the model. This again is optimal. The paper uses Leaky ReLU, however this resulted in the model not training correctly, changing the learning rate in these instances was unsuccessful.

1. Finally the dice scores are calculated.






