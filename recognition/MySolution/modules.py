class Modules:
    def __init__(self):
        self.train_size = 200
        self.kernel_initializer = "glorot_uniform"
        self.bias = True
        self.bias_initializer = "zeros"
        self.n_layers = 2
        self.layer_sizes = [32, 32]
        self.dropout = 0.5

    def get_train_size(self):
        return self.train_size

    def get_kernel_initializer(self):
        return self.kernel_initializer

    def get_bias(self):
        return self.bias

    def get_bias_initializer(self):
        return self.bias_initializer

    def get_n_layers(self):
        return self.n_layers

    def get_layer_sizes(self):
        return self.layer_sizes

    def get_dropout(self):
        return self.dropout
