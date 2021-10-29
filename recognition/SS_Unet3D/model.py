import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Input, MaxPooling3D, BatchNormalization, ReLU, concatenate, Softmax
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


# TODO: F1 Score (Dice Similarity)
# Expects Probabilities, not logits
def f1_score(y_true, y_pred):
    # print("Dice score begin")
    # print('y_pred.shape =', y_pred.shape)
    # print('y_true.shape =', y_true.shape)
    y_pred_argmax = tf.math.argmax(y_pred, axis=4)
    y_pred_onehot = to_categorical(y_pred_argmax, num_classes=6)
    dices = []
    for i in range(6):
        tmp_y_pred = y_pred_onehot[..., i]
        tmp_y_true = y_true[..., i]
        # print('y slice shape:', tmp_y_true.shape)
        # Dice!
        numerator = 2 * tf.reduce_sum(tf.multiply(tmp_y_pred, tmp_y_true)).numpy()
        # print('num:', numerator)
        denominator = tf.reduce_sum(tmp_y_pred).numpy() + tf.reduce_sum(tmp_y_true).numpy()
        # print('denom:', denominator)
        dice = numerator / denominator
        #print("dice: {}".format(dice))
        dices.append(dice)
    print('dices:', dices)
    return tf.constant(dices)


@tf.function
def weight_calc(total_samples, num_classes, current_class_samples):
    return total_samples / (num_classes * current_class_samples)

# TODO: Weighted loss so background voxels do not over-influence learning
def weighted_cross_entropy_loss(y_true, y_pred):

    class_freqs = tf.reduce_sum(y_true, axis=[0, 1, 2])  # Counts of each class in the current datum
    class_freqs = tf.cast(class_freqs, tf.float32)  # Cast to enable mapping to create weights
    n_samples = tf.reduce_sum(class_freqs)  # Sum all class frequencies to get total voxels in each channel
    n_classes = y_true.shape[-1]  # Number of classes = Number of channels (assumes channels last)

    # Weights are assigned by inverted importance based on frequency of the class's occurrence in the current datum
    # i.e. commonly occurring class is weighted low and vice versa
    class_weights = tf.map_fn(fn=lambda t: n_samples / (n_classes * t), elems=class_freqs)
    # Normalize weights to add to 1
    class_weights = tf.divide(class_weights, tf.reduce_sum(class_weights))
    # Override with hard-coded weights if required
    # class_weights = tf.constant([0, 0.5, 1, 1.5, 1.75, 2], dtype=tf.float32)

    # Create a mask based on multiplication of one-hot label and the weight, for each class, summed together.
    # When this is multiplied against the unweighted loss, the final loss is scaled (weighted) PER voxel,
    # by the class's weight factor.
    weights_mask = y_true[..., 0] * class_weights[0]
    for i in range(1, n_classes):
        weights_mask = tf.add(weights_mask, y_true[..., i] * class_weights[i])
        pass
    # Calculate unweighted Cross Entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # Return the weighted loss
    return tf.reduce_mean(weights_mask * loss)


class UNetCSIROMalePelvic:
    mdl = None
    __init = None
    __opt = None
    __loss = None
    train_batch_count = None

    # Holds a dictionary of nodes and the last node to be added to the DAG
    class ModelNodes:
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
        # Set up model parameters
        self.__init = keras.initializers.RandomNormal(stddev=0.02)
        self.__opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.__loss = tf.keras.losses.CategoricalCrossentropy()
        self.train_batch_count = 0
        # Create Model
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
                                  kernel_initializer=self.__init,
                                  padding='same')(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
                # Create Batch Normalization Layer
                new_node_name = "L{}_{}_BN_{}".format(layer_num, stage, i)
                new_node = BatchNormalization(name=new_node_name)(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
                # Create ReLU Layer
                new_node_name = "L{}_{}_ReLU_{}".format(layer_num, stage, i)
                new_node = ReLU(name=new_node_name)(mdl_nodes.last())
                mdl_nodes.add(name=new_node_name, node=new_node)
            pass

        def _analysis_block_trailer(mdl_nodes, layer_num):
            stage = stages[0]
            new_node_name = "L{}_{}_MaxPool".format(layer_num, stage)
            new_node = MaxPooling3D(name=new_node_name, pool_size=2, strides=2, padding='same')(mdl_nodes.last())
            mdl_nodes.add(name=new_node_name, node=new_node)
            pass

        def _synthesis_block(mdl_nodes, layer_num, num_maps):
            _synthesis_block_header(mdl_nodes, layer_num)
            _conv_block(mdl_nodes, layer_num, num_maps, analysis=False)
            pass

        def _synthesis_block_header(mdl_nodes, layer_num):
            stage = stages[1]
            concat_analysis_layer_name = "L{}_{}_BN_{}".format(layer_num - 1, stages[0], layer_len)
            concat_analysis_layer = mdl_nodes.find(concat_analysis_layer_name)  # Search Dict and retrieve node
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
        input_shape = (128, 128, 64, 1)
        # Create a model nodes tracker
        mdl_nodes = self.ModelNodes()
        # Input Layer
        mdl_input = Input(name="Input", shape=input_shape)
        mdl_nodes.add("Input", mdl_input)
        scale = 1
        # Build Analysis Arm of UNet
        _analysis_block(mdl_nodes, layer_num=1, num_maps=[16*scale, 32*scale])
        _analysis_block(mdl_nodes, layer_num=2, num_maps=[32*scale, 64*scale])
        _analysis_block(mdl_nodes, layer_num=3, num_maps=[64*scale, 128*scale])
        # Last layer does not have a trailer
        _conv_block(mdl_nodes, layer_num=4, num_maps=[128*scale, 256*scale])
        # Build Synthesis Arm of UNet
        _synthesis_block(mdl_nodes, layer_num=4, num_maps=[128*scale, 128*scale])
        _synthesis_block(mdl_nodes, layer_num=3, num_maps=[64*scale, 64*scale])
        _synthesis_block(mdl_nodes, layer_num=2, num_maps=[32*scale, 32*scale])
        # Final Convolution Layer
        new_node_name = "L{}_{}_FinalConv3D".format(1, 'SYN')
        new_node = Conv3D(name=new_node_name, kernel_size=1, strides=1, padding='same',
                          filters=6, activation='softmax')(mdl_nodes.last())
        mdl_nodes.add(name=new_node_name, node=new_node)
        # Instantiate & compile model object
        self.mdl = tf.keras.Model(inputs=mdl_input, outputs=mdl_nodes.last())
        self.mdl.compile(optimizer=self.__opt, loss=self.__loss, metrics=[f1_score], run_eagerly=True)
        # , tfa.metrics.F1Score(num_classes=2, threshold=0.5)],
        pass

    # TODO: Decide if required or not
    def train_batch(self, batch_size=32):
        print("Training '{}' on a batch of {}...".format(self.mdl.name, batch_size))
        return self.mdl.train_on_batch()
        pass

    # TODO: Yet to implement saving feature
    def save_model(self, model, model_series, curr_epoch, loc):
        save_name = model_series + "_" + model.name + "_AtEpoch_" + str(curr_epoch)
        model.save(r"models\{}".format(save_name))
        pass

    pass
