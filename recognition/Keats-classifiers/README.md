# COMP3710 Final Report
Knee laterality classifier based on OAI AKOA knee data using ConvNet.

As a brief overview of the problem, a dataset of some ~18k MRI images of knees was procured, I believe with the intent of training nets to detect
arthritis. I have luckily sidestepped this problem and are instead training the network to detect if it is a left or right knee, which was specified
in the path names of the images in the dataset. As an aside, I would like to point out that this neural net is nothing special. If you come up to me
and point at one of your knees, I can tell you if it is a left or right knee with at least 90% accuracy.

### Requirements
The .yaml environment has been included and it is recommended that you make an environment using this before attempting to run the driver.

### Usage
e.g. ```python3 driver.py h:/COMP3710/UnzippedKnees --relearn```
- The main file to use here is ```driver.py```, which takes some commandline arguments that it'll tell you about if you specify the -h option. 
- The most important one is the ```data_directory``` positional argument, which you should give the directory path containing the unzipped knee data. 
- An additional parameter ```--relearn``` will force the network to relearn the weights from scratch

Upon generating the model, the file will draw (using pyplot)

### Directory structure
- driver.py: The main file to run
- models.py: Contains information about the ConvNet Architecture and some functions to run once the KneeClassifier has been generated.
- images: Images of interest and used in this report
- assignment.ipynb and .py: The interactive file used before conversion to python files

### Network Architecture

![Network Structure](https://github.com/harrykeightley/PatternFlow/blob/topic-recognition/recognition/Keats-classifiers/images/layer_summary.png)


The network consists of 3 Convolutional layers, each followed by a Max pooling and using a Relu activation function. This was trial and error- it wasn't converging to a nice enough solution with 2 layers and maybe that's because the problem can't accurately be solved with only two layers, but I was also quite tired towards the end of writing this up and I may have made a mistake somewere there. For the last part, I was watching some MIT lectures and they spoke of the benefits of having a dense layer after your feature extraction layers, so I have a final hidden dense layer consisting of 100 neurons. These all connect to the final neuron which outputs the probability that the network believes the picture is a right knee.

The Activation of the final layer uses a sigmoid function to cap the probability between 0 and 1, and for the corresponding loss function it made sense to use binary_crossentropy since this is a 2 class classification problem.

The split between validation and training data was 20/80, for no real scientific reason, but because from playing around with it this seemed to give a solid amount of data to train on while not compromising the validation set. 

### Results

The network converged incredibly quickly and I actually stopped it after 10/40 epochs since it was already close to optimal from what I was seeing. Images of  classifications performed on random validation_set data by the network has been included below. __p__ refers to predicted and __t__ to true labels. The accuracy it gave on the validation data converged quite quickly to 100%, which is cool but I still feel like that's a little too high and that I did something wrong.

![Knee Data](https://github.com/harrykeightley/PatternFlow/blob/topic-recognition/recognition/Keats-classifiers/images/knee_data.png)

