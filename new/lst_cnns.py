import keras,os
import densenet
from keras.models import Sequential, Model
from keras.layers import Add, Input, ZeroPadding2D, Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Activation, AveragePooling2D
from keras.regularizers import l2
from keras.initializers import he_uniform, glorot_normal, he_uniform, he_normal
#from keras.preprocessing.image import ImageDataGenerator
#import numpy as np

'''
class LST_DenseNet:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = densenet.DenseNet(
            include_top=False,
            input_tensor=input_img,
            depth=100,
            growth_rate=24,
            bottleneck=True,
            reduction=0.5
        )
        x = base.layers[-1].output
        # x = Flatten(name='flatten')(x)
        # x = Dense(1000, activation='linear', name='fc1', kernel_initializer='he_normal')(x)
        x = Dense(self.outcomes, name='regression', activation='linear', kernel_initializer='he_normal')(x)
        model = Model(inputs=input_img, output=x, name="DenseNet")

        return model
'''

class LST_VGG16:

    def __init__(
            self,
            channels,
            img_rows,
            img_cols,
            outcomes=1,
            last_activation='linear',
            dropout_rate=0.5,
            weight_decay=1e-4
    ):
        self.outcomes = outcomes
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()
        self.dropout = dropout_rate
        self.wd = weight_decay
        self.last = last_activation
        self.init = 'he_uniform'

    def get_model(self):

        self.model.add(BatchNormalization(input_shape=(self.img_rows, self.img_cols, self.channels)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3),
                              padding="same", activation="relu", kernel_initializer=self.init,
                              kernel_regularizer=l2(self.wd)))
        #self.model.add(Conv2D(input_shape=(self.img_rows, self.img_cols, self.channels), filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Dropout(self.dropout))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Dropout(self.dropout))

        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(self.dropout))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(self.dropout))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(self.dropout))

        self.model.add(Flatten())
        #self.model.add(Dense(units=4096, activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        #self.model.add(Dropout(self.dropout))
        #self.model.add(Dense(units=4096, activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        #self.model.add(Dropout(self.dropout))
        self.model.add(Dense(units=self.outcomes, activation=self.last, kernel_initializer=self.init))

        return self.model

################################################################################################################################
##################################################   ResNets   #################################################################
################################################################################################################################

# define identity block and convolutional block
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_normal(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_normal(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# define ResNets of different depths
class LST_ResNet50:

    def __init__(
            self,
            channels,
            img_rows,
            img_cols,
            outcomes=1,
            last_activation='linear',
            dropout_rate=0.5,
            weight_decay=1e-4
    ):
        self.outcomes = outcomes
        self.channels = channels
        self.input_shape = (img_rows, img_cols, channels)
        self.dropout = dropout_rate
        self.wd = weight_decay
        self.last = last_activation
        #self.init = 'he_uniform'
        #self.init = 'he_normal'
        #self.init = 'glorot_uniform'
        #self.init = 'glorot_normal'

    #def get_model(input_shape=(64, 64, 3), classes=6):
    def get_model(self):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(self.input_shape)
        # X = BatchNormalization(axis=3, name='initial_bn')(X_input)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_normal(seed=0))(X)
        #X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dense(self.outcomes, activation=self.last, name='fc' + str(self.outcomes), kernel_initializer=glorot_normal(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model