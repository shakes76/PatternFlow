import tensorflow as tf

class YoloV1():

    def __init__(self, imageWidth=488, imageHeight=488, S=7, B=2, C=1, lambdaCoord=5, lambdaNoObj=0.5):
        """Create a new YOLOV1 instance.

        Args:
            imageWidth (int) : The width of the datasets images. Once the model has been trained on this size, this cannot change.
            imageHeight (int) : The height of the datasets images. Once the model has been trained on this size, this cannot change. 
            S (int): The number of cells the images row and colums should be divided into, where S*S is the total number. 
            B (int): The number of bounding boxes to be predicted per cell.
            C (int): The number of classes in the entire dataset. 
            lambdaCoord (int): An int to multiply against various individual loss calculations. Used to prioritise (increase) certain losses.
            lambdaNoObj (int): An int to multiply against the no class loss. Used to decrease the priority of this loss. 
        """
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.S = S
        self.B = B
        self.C = C
        self.lambdaCoord = lambdaCoord
        self.lambdaNoObj = lambdaNoObj
        self.model = self.modelArchitecture()

    def jaccardIndex(self, y_true, y_pred):
        """A custom metric for the model. Calculates the jaccard index. 
        Args:
            y_true (tensorflow.DataSet): A tensor containing the true bounding boxes. Shape (batchSize, S, S, B*5+C). 
            y_pred (tensorflow.DataSet): A tensor containing the predicted bounding boxes. Shape (batchSize, S, S, 5+C)
        Returns:
            tensorflow.Constant: A tensor containing the jaccardIndex between the best box and true box.
        """
        y_true = tf.reshape(y_true[...,:5], [-1,self.S*self.S,1,5])
        y_pred = tf.reshape(y_pred[...,:self.B*5], [-1,self.S*self.S,self.B,5])
        
        # Get the true bounding box. 
        y_true_loc = tf.where(tf.equal(y_true[...,0], tf.math.reduce_max(y_true[...,0])))[0]
        y_true_box = tf.gather_nd(y_true, y_true_loc)
        # Get the bounding box that has the highest confidence
        y_pred_loc = (tf.where(tf.equal(y_pred[...,0], tf.math.reduce_max(y_pred[...,0]))))[0]
        y_pred_box = tf.gather_nd(y_pred, y_pred_loc)

        # Convert the boxes to a standard box format
        yMinT, xMinT, yMaxT, xMaxT = self.convertYoloBoxToStandard(y_true_box, y_true_loc) 
        yMinP, xMinP, yMaxP, xMaxP = self.convertYoloBoxToStandard(y_pred_box, y_pred_loc)

        # Determine the coordinates of the intersection area
        maxValue = lambda x,y: x if tf.math.greater(x,y) else y
        xMin = maxValue(xMinT, xMinP)
        yMin = maxValue(yMinT, yMinP)
        xMax = maxValue(xMaxT, xMaxP)
        yMax = maxValue(yMaxT, yMaxP)
        # Caculate the area of the intersection
        interArea = (xMax - xMin) * (yMax - yMin)
        if interArea < 0.0:
            interArea = tf.zeros_like(interArea)
        # Compute area of both boxes
        trueArea = (xMaxT-xMinT) * (yMaxT-yMinT)
        predArea = (xMaxP-xMinP) * (yMaxP-yMinP)
        denom = (trueArea + predArea - interArea)
        if tf.equals(denom, 0.0):
            return tf.zeros_like(denom)
        # Calculate IOU
        iou = interArea / (trueArea + predArea - interArea)
        
        return iou

    def convertYoloBoxToStandard(self, box, boxGridLocation):
        """Convert a box of the format (centerX, centerY, boxWidth, boxHeight) -> true pixel coords (xMin, yMin, xMax, yMax)
        Where centerX, centerY are relative to cell size, and boxWidth, boxHeight are relative too image size.

        Args: 
            box (tensorflow.Tensor): A single bounding box, shape (5,)
            boxGridLocation: The location of box in the grid,  shape (4,)
        Returns:
            xMin (tensorflow.Constant): The minimum true pixel x coord of the bounding box
            yMin (tensorflow.Constant): The minimum true pixel y coord of the bounding box
            xMax (tensorflow.Constant): The maximum true pixel x coord of the bounding box
            yMax (tensorflow.Constant): The maximum true pixel y coord of the bounding box
        """
        boxWidth = tf.cast(box[3], tf.float32) * tf.cast(self.imageWidth/2, tf.float32)
        boxHeight = tf.cast(box[4], tf.float32) * tf.cast(self.imageHeight/2, tf.float32)
        boxCenterX = tf.cast(box[1], tf.float32) * tf.cast(self.imageWidth/self.S, tf.float32) + tf.cast(boxGridLocation[1], tf.float32)*tf.cast(self.imageWidth/self.S, tf.float32)
        boxCenterY = tf.cast(box[2], tf.float32) * tf.cast(self.imageHeight/self.S, tf.float32) + tf.cast(boxGridLocation[2], tf.float32) *tf.cast(self.imageHeight/self.S, tf.float32)
        nonZero = lambda x: x if x > 0.0 else tf.zeros_like(x) 
        noOverFlow = lambda x: x if x < tf.cast(self.imageWidth, tf.float32) else tf.cast(self.imageWidth, tf.float32)
        xMin = nonZero(boxCenterX - boxWidth)
        xMax = noOverFlow(boxCenterX + boxWidth)
        yMin = nonZero(boxCenterY - boxHeight)
        yMax = noOverFlow(boxCenterY + boxHeight)

        return tf.math.round(yMin), tf.math.round(xMin), tf.math.round(yMax), tf.math.round(xMax)

    def yoloLoss(self, y_true, y_pred):
        """A custom loss function as specificied by the yolov1 paper. 
        Args:
            y_true (tensorflow.DataSet): A tensor containing the true bounding boxes. Shape (batchSize, S, S, B*5+C). 
            y_pred (tensorflow.DataSet): A tensor containing the predicted bounding boxes. Shape (batchSize, S, S, 5+C)
        Returns:
            tensorflow.Dataset: A tensor containing the loss for each cell. Will be of shape (batchsize, S*S).
        """
        true_boxes = tf.reshape(y_true[...,:5], [-1,self.S*self.S,1,5])
        pred_boxes = tf.reshape(y_pred[...,:self.B*5], (-1,self.S*self.S,self.B,5))
        # Reshape the 3: too S*S, B, 5 i.e. 49, 2, 5. 
        # Each cell has two bounding boxes, with each bounding box, having 5 elements. 
        
        # Calculate the IOU for every box.
        num = tf.math.multiply(true_boxes[...,1:5], pred_boxes[...,1:5])
        denom = true_boxes[...,1:5] + pred_boxes[...,1:5] - num
        iou = tf.math.reduce_sum(num/denom,  axis=-1)

        # If boxes in the same cell have the same IOU, add 1 to the first box (since they are both equal) 
        duplicates = tf.where(tf.equal(iou[...,0], iou[...,1]))
        iou = tf.tensor_scatter_nd_add(iou, duplicates, tf.add(tf.zeros_like(duplicates, dtype=tf.float32), [1,0]))
        
        # Get the indices of the max IOUs for each cell and grab the respetive boxes.
        maxIou = tf.math.reduce_max(iou, axis=-1, keepdims=True)
        maxIdx = tf.where(tf.equal(iou, maxIou))
        # Use the best boxes for calculations in the xy_loss, wh_loss, and conf_loss (as per the formula)
        best_boxes = tf.reshape(tf.gather_nd(pred_boxes, maxIdx), [-1, self.S*self.S,1,5])

        # The confidence on whether there is an object in the cell, between 0-1
        y_pred_conf = best_boxes[...,0]
        y_true_conf = true_boxes[...,0]
        y_true_conf_noob = tf.math.subtract(tf.ones_like(y_true_conf), y_true_conf)
        
        # The width and height of the bounding boxes, normalised to the image size
        y_pred_wh   = best_boxes[...,3:5]
        y_true_wh   = true_boxes[...,3:5]

        # The centre of the bounding box, normalised to the cell size
        y_pred_xy   = best_boxes[...,1:3]
        y_true_xy   = true_boxes[...,1:3]
        
        # The classes that are within this cell. 0 if  no, 1 if yes. Noting that there is only one class in this instance. 
        y_true_class = tf.reshape(y_true[...,5:], [-1, self.S*self.S, self.C]) 
        y_pred_class = tf.reshape(y_pred[...,self.B*5:], [-1, self.S*self.S, self.C])
        
        # Losses Calculations +++++++++++++++++++++++++++++++++++++++++++++++++++
        #y_true_conf will 0 cells that do not have an object in the ground truth 
        xy_loss = self.lambdaCoord * tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(y_true_xy - y_pred_xy),
                                                        axis=-1)*y_true_conf, axis=-1)
    
        
        # Two reduce sums  = 2 summations. Axis = -1, along last dimensions i.e. columns/entries in this case.
        # Square is element wise. y_true_conf will 0 cells that do not have an object in the ground truth 
        wh_loss = self.lambdaCoord * tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(tf.math.sqrt(y_true_wh) - tf.math.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)      

        #y_true_conf will 0 cells that do not have an object in the ground truth 
        #y_true_conf_noob will 0 cells that DO have an object in the ground truth 
        conf_loss = tf.math.reduce_sum(tf.math.square(y_true_conf - y_pred_conf)*y_true_conf, axis=-1)
        conf_loss_noob = self.lambdaNoObj * tf.math.reduce_sum(tf.math.square(y_true_conf - y_pred_conf)*y_true_conf_noob, axis=-1)

        clss_loss  = tf.math.reduce_sum(tf.math.square(y_true_class - y_pred_class)*y_true_conf, axis=-1) 

        loss =  clss_loss + xy_loss + wh_loss + conf_loss + conf_loss_noob

        return loss

    def modelArchitecture(self):
        """Defines the YoloV1 neural network. 
        Some additions have been made to the original architecture. They are follows: 
            1) Batch Normalization has been introduced to make the network more stable and increase peformance.
            2) A sigmoid activation function has been introduced to speed up training time (over linear activation). 
                -- Removed due to network instability

        Returns:
            tensorflow.keras.Sequential: A Convolutional Neural Network defined by the YoloV1 architecture.  
        """
        model = tf.keras.Sequential([
            #First Layer
            tf.keras.layers.Conv2D(64, (7,7), strides=(2, 2),  input_shape=(self.imageWidth,self.imageHeight,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Second Layer
            tf.keras.layers.Conv2D(192, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Third Layer    
            tf.keras.layers.Conv2D(128, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Fourth Layer
            # +++ Repeated block
            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # +++ END BLOCK
            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Fifth layer
            # +++ Repeated Block
            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # +++ END BLOCK
            tf.keras.layers.Conv2D(1024, (3,3),  strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            #Sixth Layer
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            # Final Output Layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(self.S * self.S * (self.B*5+self.C), input_shape=(4096,)),
            tf.keras.layers.Reshape(target_shape = (self.S, self.S, (self.B*5+self.C)))
        ])

        return model

    def compileModel(self, learning_rate=0.001, clipnorm=1.0, run_eagerly=False, **kwargs):
        """Compiles self.model using the yoloLoss and jaccardIndex.

        Args:
            learning_rate (int): An int that specifies the learning rate to train the model with. 
            clipnorm (clipnorm): An int that speicifies the amount the gradient should limited to be between. 
            run_eagerly (boolean): Whether or not the model should be run eagerly, useful for debugging. 
            **kwargs (dict): Any additional parameters to be passed to the model. 
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
              loss=self.yoloLoss, metrics=[self.jaccardIndex], run_eagerly=run_eagerly, **kwargs)

    def loadWeights(self, checkpoint):
        """Loads existing weights into the model. 
        Useful for predicting new images, or training with additional data.

        Args:
            checkPoint (str): The path to the checkpoint file. 
        """
        self.model.load_weights(checkpoint)

    def predictData(self, testData):
        """Predicts a bounding box for an image. 
        Args: 
            testData (tensorflow.Dataset): The data on which the bounding boxes are too be predicted.
        Returns:
            tensorflow.Tensor: A tensor containing the predicted bounding boxes. 
        """
        return self.model.predict(testData)

    def runModel(self, train_batches, validation_batches, epochs=80):
        """Using model.fit, runs yolov1 model on the provided data.

        Args:
            train_batches (tensorflow.DataSet): A dataset containing the batches of training data.
            validation_batches (tensorflow.DataSet): A dataset containing the batches of validation data.
            epochs (int): The number of epochs for the model. 
        Returns:
            tensorflow.History: a record of training loss values and metrics values at successive epochs, 
                as well as validation loss values and validation metrics values.
        """
        self.history = self.model.fit(train_batches, epochs=epochs, validation_data=validation_batches)
        return self.history