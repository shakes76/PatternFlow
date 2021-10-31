class EarlyStopping():
    def __init__(self, patience=5, tol=0.05):
        self.patience = patience
        self.tol = tol
        self.lowest_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, test_loss):

        if self.lowest_loss == None:
            # for the first iteration
            self.lowest_loss = test_loss
        elif self.lowest_loss - test_loss > self.tol:
            # if the current loss is smaller than last iteration
            # then counter set to 0
            # loss may be fluctuation
            self.lowest_loss = test_loss
            self.counter = 0
        elif self.lowest_loss - test_loss < self.tol:
            # if the current loss is grater than last iteration
            # then counter+1
            # the loss should be keep decreasing
            self.counter += 1
            print("Early stopping in {} /{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('Early Stops')
                self.early_stop = True
        return self.early_stop
