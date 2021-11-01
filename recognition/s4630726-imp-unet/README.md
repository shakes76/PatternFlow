IMPROVED UNET 
Welcome to the improved UNET!

The improved unet utilises several different changes from the one in demo 2. For example, the verboseness is reduced by the means of simplifying repeated patterns of convolutions and other keras operations into individual modules that be easy to add when one of the modules pop up. Another change is the structure towards the end where the individal localization output are summed together before going through the activation layer.

To use this version of improved unet. First make sure you have the ISIC data set found at https://cloudstor.aarnet.edu.au/sender/?s=download&token=723595dd-15b0-4d1e-87b8-237a7fe282ff (delete the txt files at the beginning and end)

you will also need to make sure the following dependencies are installed being:


-Anaconda3 (or the substituents used in the code that are Anaconda3)
-tensorflow
-os for directory operations
-cv2 for image reading
-scikit for splitting the data into training and test sets
-scikit image ofr image manipulation (resizing etc)

Then open up main.py in your text editor and change the "root" variable to where ever the ISIC data is stored. In that directory there should be two folders: 
-"ISIC2018_Task1-2_Training_Input_x2"
-"ISIC2018_Task1_Training_GroundTruth_x2"

Then using your console navigate to where the directory s4630726-imp-unet is located

type in the console "python main.py" and then the script will run all the way through untill the end where the DSC scores are calculated for each class 

RESULTS: 
the model definitely works and segements moles, some very accurately. Some not so good. This is unfortuneately reflected in the DSC score where the best I was alble to get was ~0.76. Not able to achieve the goal of 0.8 DSC for each class

CHANGES MADE TO ACHIEVE RESULTS AND HYPERPARAMETERS:
the biggest notable change happened afer changing from grayscale a implementation to a colour one (GBR). The DSC scores went from ~0.71-0.72 to ~0.75-0.76
activation function: sigmoid
loss function: DSC
learning rate (stastic): 1e-5
threshold for predictions: 0.91
epochs: 100 (for best results)
batch size: 20
optimizer: Adam

The paper uses LeakyReLU, however this resulted in the model not training correctly, changing the learning rate in these instances was unsuccessful 




