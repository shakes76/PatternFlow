import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Dropout, Input, MaxPooling3D, BatchNormalization, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import concatenate


class UNetCSIROMalePelvic:
    mdl = None
    init = keras.initializers.RandomNormal(stddev=0.02)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss = tf.keras.losses.binary_crossentropy

    # Holds a dictionary of nodes and the last node to be added to the DAG
    class ModelNodes():
        nodes = [{}, None]
        def __init__(self):
            pass

        def find(self, node):
            return self.nodes[0][node]

        def last(self):
            return self.nodes[1]

        def add(self, name, node):
            self.nodes[0][name] = node
            self.nodes[1] = node

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
                new_node = Conv3D(name=new_node_name, kernel_size=3, filters=num_maps[i-1],
                                  activation='relu', kernel_initializer=self.init,
                                  padding='same')(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
                # Create Batch Normalization Layer
                new_node_name = "L{}_{}_BN_{}".format(layer_num, stage, i)
                new_node = BatchNormalization(name=new_node_name)(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
            pass

        def _analysis_block_trailer(mdl_nodes, layer_num):
            stage = stages[0]
            new_node_name = "L{}_{}_MaxPool".format(layer_num, stage)
            new_node = MaxPooling3D(name=new_node_name, pool_size=2, strides=2,padding='same')(mdl_nodes.last())
            mdl_nodes.add(name=new_node_name, node=new_node)
            pass

        def _synthesis_block(mdl_nodes, layer_num, num_maps):
            _synthesis_block_header(mdl_nodes, layer_num)
            _conv_block(mdl_nodes, layer_num, num_maps, analysis=False)
            pass

        def _synthesis_block_header(mdl_nodes, layer_num):
            stage = stages[1]
            concat_analysis_layer_name = "L{}_{}_BN_{}".format(layer_num - 1, stages[0], layer_len)
            concat_analysis_layer = mdl_nodes.find(concat_analysis_layer_name)  # Search Dict and retrive node
            last_layer_filters = mdl_nodes.last().shape[-1]
            # Step 1 - De-convolution while maintaining same number of filters
            new_node_name = "L{}_{}_Conv3DTranspose".format(layer_num, stage)
            new_node = Conv3DTranspose(name=new_node_name, kernel_size=2, strides=2,
                                       padding='same', filters=last_layer_filters)(mdl_nodes.last())
            mdl_nodes.add(name=new_node_name, node=new_node)
            # Step 2 - Concatenate with Analysis Conjugate
            new_node_name = "L{}_{}_Concat".format(layer_num, stage)
            new_node = concatenate(name=new_node_name, inputs=[mdl_nodes.last(), concat_analysis_layer])
            mdl_nodes.add(name=new_node_name, node=new_node)
            pass

        def _analysis_block(mdl_nodes, layer_num, num_maps):
            _conv_block(mdl_nodes, layer_num, num_maps, analysis=True)
            _analysis_block_trailer(mdl_nodes, layer_num)
            pass

        # ==========================================================================================
        # Begin creating model
        input_shape = (256, 256, 128, 1)
        # Create a model nodes tracker
        mdl_nodes = self.ModelNodes()
        # Input Layer
        mdl_input = Input(name="Input", shape=input_shape)
        mdl_nodes.add("Input", mdl_input)
        # Begin Analysis Arm of UNet
        _analysis_block(mdl_nodes, layer_num=1, num_maps=[32, 64])
        _analysis_block(mdl_nodes, layer_num=2, num_maps=[64, 128])
        _analysis_block(mdl_nodes, layer_num=3, num_maps=[128, 256])
        # Last layer does not have a trailer
        _conv_block(mdl_nodes, layer_num=4, num_maps=[256, 512])
        # Begin Synthesis Arm of UNet
        _synthesis_block(mdl_nodes, layer_num=4, num_maps=[256, 256])
        _synthesis_block(mdl_nodes, layer_num=3, num_maps=[128, 128])
        _synthesis_block(mdl_nodes, layer_num=2, num_maps=[64, 64])
        # Instantiate & compile model object
        self.mdl = tf.keras.Model(inputs=mdl_input, outputs=mdl_nodes.last())
        self.mdl.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'])
        pass

    def train_batch(self, batch_size=32):
        print("Training '{}' on a batch of {}...".format(self.mdl.name, batch_size))
        pass

    def save_model(self, loc):
        # Save Model
        pass

    pass
