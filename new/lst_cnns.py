import keras,os
import densenet
from keras.models import Sequential, Model
from keras.layers import Add, Input, ZeroPadding2D, Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Activation, AveragePooling2D, Reshape, multiply, GlobalAveragePooling2D, Permute
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
            dropout_rate=0.2,
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
        n_filters = 64
        #1,2
        self.model.add(Conv2D(filters=n_filters, kernel_size=(3, 3),
                              padding="same", activation="relu", kernel_initializer=self.init,
                              kernel_regularizer=l2(self.wd)))
        #self.model.add(Conv2D(input_shape=(self.img_rows, self.img_cols, self.channels), filters=64, kernel_size=(3, 3),padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))

        self.model.add(Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Dropout(self.dropout))

        #self.model.add(BatchNormalization())
        #3,4
        self.model.add(Conv2D(filters=2*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=2*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Dropout(self.dropout))

        #5,6,7
        #self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=4*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=4*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=4*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        #8,9,10
        self.model.add(Dropout(self.dropout*3./5.))
        #self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Dropout(self.dropout))
        #self.model.add(BatchNormalization())
        #11,12,13
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Conv2D(filters=8*n_filters, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        #final block
        #self.model.add(Dropout(self.dropout))
        #self.model.add(BatchNormalization())
        self.model.add(Flatten())
        #self.model.add(Dense(units=32, activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Dropout(self.dropout))
        #self.model.add(BatchNormalization())
        #self.model.add(Dense(units=128, activation="relu", kernel_initializer=self.init, kernel_regularizer=l2(self.wd)))
        self.model.add(Dense(units=self.outcomes, activation=self.last, kernel_initializer=self.init))

        return self.model

################################################################################################################################
##################################################   ResNets   #################################################################
################################################################################################################################

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if keras.backend.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if keras.backend.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

# define identity block and convolutional block
def identity_block(X, f, filters, stage, block, k_init='he_uniform', se=False, wd=1e-4):
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
               kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    if se:
        X = squeeze_excite_block(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2, k_init='he_uniform', se=False, wd=1e-4):
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
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=k_init, kernel_regularizer=l2(wd))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=k_init, kernel_regularizer=l2(wd))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    if se:
        X = squeeze_excite_block(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# define ResNets of different depths
class LST_ResNet18:

    def __init__(
            self,
            channels,
            img_rows,
            img_cols,
            outcomes=1,
            last_activation='linear',
            dropout_rate=0.5,
            weight_decay=1e-4,
            se=False
    ):
        self.outcomes = outcomes
        self.channels = channels
        self.input_shape = (img_rows, img_cols, channels)
        self.dropout = dropout_rate
        self.wd = weight_decay
        self.last = last_activation
        self.init = 'he_uniform'
        self.se = se
        #self.init = 'he_normal'
        #self.init = 'glorot_uniform'
        #self.init = 'glorot_normal'

    def i_block(self, X, f, filters, stage, block):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a',
                   kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)


        if self.se:
            X = squeeze_excite_block(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def c_block(self, X, f, filters, stage, block, s=2):
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
        F1, F2 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a',
                   kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='same', name=conv_name_base + '1',
                            kernel_initializer=self.init)(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        if self.se:
            X = squeeze_excite_block(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    #def get_model(input_shape=(64, 64, 3), classes=6):
    def get_model(self):
        """
        Implementation of the popular ResNet18 the following architecture:
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
        X = BatchNormalization(axis=3, name='initial_bn')(X_input)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X)

        # Stage 1
        n1 = 16
        n2 = 16
        X = Conv2D(n1, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.c_block(X, f=3, filters=[n1, n2], stage=2, block='a', s=1)
        X = self.i_block(X, 3, [n1, n2], stage=2, block='b')

        ### START CODE HERE ###
        n1 *= 2
        n2 *= 2
        # Stage 3 (≈4 lines)
        X = self.c_block(X, f=3, filters=[n1, n2], stage=3, block='a', s=2)
        X = self.i_block(X, 3, [n1, n2], stage=3, block='b')

        # Stage 4 (≈6 lines)
        n1 *= 2
        n2 *= 2
        X = self.c_block(X, f=3, filters=[n1, n2], stage=4, block='a', s=2)
        X = self.i_block(X, 3, [n1, n1], stage=4, block='b')
        X = Dropout(self.dropout)(X)

        # Stage 5 (≈3 lines)
        n1 *= 2
        n2 *= 2
        X = self.c_block(X, f=3, filters=[n1, n2], stage=5, block='a', s=2)
        X = self.i_block(X, 3, [n1, n2], stage=5, block='b')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dropout(self.dropout)(X)
        X = Dense(self.outcomes, activation=self.last, name='fc' + str(self.outcomes), kernel_initializer=self.init)(X)

        # Create model
        if self.se:
            model = Model(inputs=X_input, outputs=X, name='ResNet18SE')
        else:
            model = Model(inputs=X_input, outputs=X, name='ResNet18' )

        return model
'''

class LST_ResNet18:

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
        self.init = 'he_uniform'
        #self.init = 'he_normal'
        #self.init = 'glorot_uniform'
        #self.init = 'glorot_normal'

        def block(n_output, upscale=False):
            # n_output: number of feature maps in the block
            # upscale: should we use the 1x1 conv2d mapping for shortcut or not

            # keras functional api: return the function of type
            # Tensor -> Tensor
            def f(x):

                # H_l(x):
                # first pre-activation
                h = BatchNormalization()(x)
                h = Activation('relu')(h)
                # first convolution
                h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                           kernel_regularizer=l2(self.wd))(h)

                # second pre-activation
                h = BatchNormalization()(x)
                h = Activation('relu')(h)
                # second convolution
                h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                           kernel_regularizer=l2(self.wd))(h)

                # f(x):
                if upscale:
                    # 1x1 conv2d
                    f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
                else:
                    # identity
                    f = x

                # F_l(x) = f(x) + H_l(x):
                return Add()([f, h])

            return f

        def get_model(self):
            # input tensor is the 28x28 grayscale image
            X_input = Input(self.input_shape)

            # first conv2d with post-activation to transform the input data to some reasonable form
            x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=l2(self.wd))(X_input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # F_1
            x = block(16)(x)
            # F_2
            x = block(16)(x)

            # F_3
            # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
            # and we can't add together tensors of inconsistent sizes, so we use upscale=True
            # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
            # F_4
            # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
            # F_5
            # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

            # F_6
            # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
            # F_7
            # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

            # last activation of the entire network's output
            x = BatchNormalization()(x)
            x = Activation(relu)(x)

            # average pooling across the channels
            # 28x28x48 -> 1x48
            x = GlobalAveragePooling2D()(x)

            # dropout for more robust learning
            x = Dropout(0.2)(x)

            # last softmax layer
            x = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)
            x = Activation(softmax)(x)

            # Create model
            model = Model(inputs=X_input, outputs=X, name='ResNet50SE')

            return model
'''
'''
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
        self.init = 'he_uniform'
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
        X = BatchNormalization(axis=3, name='initial_bn')(X_input)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X)

        # Stage 1
        n1 = 16
        n2 = 4*n1
        X = Conv2D(32, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=2, block='a', s=1, se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=2, block='b', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=2, block='c', se=True)

        ### START CODE HERE ###
        n1 *= 2
        n2 *= 2
        # Stage 3 (≈4 lines)
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=3, block='a', s=2, se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='b', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='c', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='d', se=True)

        # Stage 4 (≈6 lines)
        n1 *= 2
        n2 *= 2
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=4, block='a', s=2, se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='b', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='c', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='d', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='e', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='f', se=True)

        # Stage 5 (≈3 lines)
        n1 *= 2
        n2 *= 2
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=5, block='a', s=2, se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=5, block='b', se=True)
        X = identity_block(X, 3, [n1, n1, n2], stage=5, block='c', se=True)

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dense(self.outcomes, activation=self.last, name='fc' + str(self.outcomes), kernel_initializer=self.init)(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model
'''
class LST_ResNet50:

    def __init__(
            self,
            channels,
            img_rows,
            img_cols,
            outcomes=1,
            last_activation='linear',
            dropout_rate=0.5,
            weight_decay=1e-4,
            se=False
    ):
        self.outcomes = outcomes
        self.channels = channels
        self.input_shape = (img_rows, img_cols, channels)
        self.dropout = dropout_rate
        self.wd = weight_decay
        self.se = se
        self.last = last_activation
        self.init = 'he_uniform'
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
        X = BatchNormalization(axis=3, name='initial_bn')(X_input)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X)

        # Stage 1
        n1 = 32
        n2 = 4*n1
        X = Conv2D(32, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=self.init)(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=2, block='a', s=1, wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=2, block='b', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=2, block='c', wd=self.wd, se=self.se)

        ### START CODE HERE ###
        n1 *= 2
        n2 *= 2
        # Stage 3 (≈4 lines)
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=3, block='a', s=2, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='b', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='c', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=3, block='d', wd=self.wd, se=self.se)

        # Stage 4 (≈6 lines)
        n1 *= 2
        n2 *= 2
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=4, block='a', s=2, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='b', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='c', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='d', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='e', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=4, block='f', wd=self.wd, se=self.se)

        # Stage 5 (≈3 lines)
        n1 *= 2
        n2 *= 2
        X = Dropout(self.dropout)(X)
        X = convolutional_block(X, f=3, filters=[n1, n1, n2], stage=5, block='a', s=2, wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=5, block='b', wd=self.wd, se=self.se)
        X = identity_block(X, 3, [n1, n1, n2], stage=5, block='c', wd=self.wd, se=self.se)

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dropout(self.dropout)(X)
        X = Dense(self.outcomes, activation=self.last, name='fc' + str(self.outcomes), kernel_initializer=self.init)(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50SE')

        return model