import model

def main():
    """ running this file with start training the GCN model for the facebook dataset.
        After training is done it will evaluate the validation accuracy.
        After the validation accuracy is evaluated it will graph the validation
        accuracy along with its loss and also the training accuracy and loss.
        Lastly it will give a TSNE plot of the dataset and how it was evaluated given
        specific colours. If there are more than 4 colours there was an error.
        Also note that the TSNE plot might not match the colours given in the
        README file since the models changes each time it is run and the colour
        choices are random.
    """
    model.run_model()
  

if __name__=="__main__":
    main()