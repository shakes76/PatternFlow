import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Dropout, Input, MaxPooling3D, BatchNormalization, Concatenate, concatenate
from tensorflow.keras.models import Sequential
from tensorflow import keras


class UNetCSIROMalePelvic:
    mdl = None
    init = keras.initializers.RandomNormal(stddev=0.02)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss = tf.keras.losses.binary_crossentropy


    class ModelNodes():
        nodes = [{}, []]
        def __init__(self):
            pass

        def last(self):
            return self.nodes[1][-1]

        def add(self, name, node):
            self.nodes[0][name] = node
            self.nodes[1].append(node)




    def __init__(self, given_name):
        # Set up model
        self._create_model(given_name)

    def _create_model(self, given_name):
        layer_len = 2
        stages = ['ANL', 'SYN']

        def _conv_block(mdl_nodes, layer_num, num_maps, analysis=True):
            stage = stages[0] if analysis else stages[1]
            for i in range(1, layer_len + 1):
                # Create Conv3D Layer
                new_node_name = "L{}_{}_Conv3D_{}".format(layer_num, stage, i)
                new_node = Conv3D(name=new_node_name, kernel_size=3, filters=num_maps*i,
                                  activation='relu', kernel_initializer=self.init,
                                  padding='same')(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
                # Create Batch Normalization Layer
                new_node_name = "L{}_{}_BN_{}".format(layer_num, stage, i)
                new_node = BatchNormalization(name=new_node_name)(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)

        def _analysis_block_trailer(mdl_nodes, layer_num):
            stage = stages[0]
            new_node_name = "L{}_{}_MaxPool".format(layer_num, stage)
            new_node = MaxPooling3D(name=new_node_name, pool_size=2, strides=2)(mdl_nodes.last())
            mdl_nodes.add(name=new_node_name, node=new_node)

        def _synthesis_block_header(given_mdl, layer_num):
            stage = stages[1]
            fetch_layer = given_mdl.get_layer("L{}_{}_BN_{}".format(layer_num - 1, stages[0], layer_len))
            #fetch_layer_filters = fetch_layer.output.shape[-1]
            last_layer_filters = given_mdl.layers[-1].output.shape[-1]
            given_mdl = Conv3DTranspose(kernel_size=2, strides=2, filters=last_layer_filters,
                                          name="L{}_{}_Conv3DTranspose".format(layer_num, stage))(given_mdl)
            # Concatenate
            given_mdl = tf.keras.layers.Concatenate()([
                    given_mdl.layers[-1].output,
                    fetch_layer.output
                    ])(given_mdl)
            return given_mdl


            #print('Test', fetch_layer_filters)
            #print(fetch_layer.name)


        def _analysis_block(mdl_nodes, layer_num, num_maps):
            _conv_block(mdl_nodes, layer_num, num_maps, analysis=True)
            _analysis_block_trailer(mdl_nodes, layer_num)

        '''
        self.mdl = Sequential(name=given_name)
        self.mdl.add(Input(shape=(256, 256, 128, 1)))

        _analysis_block(self.mdl, layer_num=1, num_maps=32)
        _analysis_block(self.mdl, layer_num=2, num_maps=64)
        _analysis_block(self.mdl, layer_num=3, num_maps=128)
        # Last layer does not have a trailer
        _conv_block(self.mdl, layer_num=4, num_maps=256)
        # Begin Synthesis Arm
        _synthesis_block_header(self.mdl, layer_num=4)
        '''

        mdl_nodes = self.ModelNodes()  # [{},list()]

        mdl_input = Input(name="Input", shape=(256, 256, 128, 1))
        mdl_nodes.add("Input", mdl_input)

        mdl_output = _analysis_block(mdl_nodes, layer_num=1, num_maps=32)

        #mdl_output = _analysis_block(mdl_output, layer_num=2, num_maps=64)
        #mdl_output = _analysis_block(mdl_output, layer_num=3, num_maps=128)
        # Last layer does not have a trailer
        #mdl_output = _conv_block(mdl_output, layer_num=4, num_maps=256)
        # Begin Synthesis Arm
        #mdl_output = _synthesis_block_header(mdl_output, layer_num=4)


        self.mdl = tf.keras.Model(inputs=mdl_input, outputs=mdl_nodes.last())
        #print(mdl_output)


        # Define the Layers
        # Define the Layers
        # ...
        # ...
        # Define the Layers
        # Define the Layers
        #self.mdl.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'])
        #self.mdl.build()
        pass

    def train_batch(self, batch_size=32):
        print("Training '{}' on a batch of {}...".format(self.mdl.name, batch_size))
        pass

    def save_model(self, loc):
        # Save Model
        pass

    pass
