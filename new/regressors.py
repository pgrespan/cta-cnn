import warnings
warnings.simplefilter('ignore')

import keras
from keras import layers, models
from keras import backend as K
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D, BatchNormalization, Activation, Input, \
    Reshape, multiply, GlobalAveragePooling2D, Permute, MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2

import densenetlst
# import resnext

class ResNet50:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="resnet50")

        return model

class RegressorV2:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        cf = 'channels_first'
        ishape = (self.channels, self.img_rows, self.img_cols)

        self.model.add(Conv2D(16, kernel_size=(3, 3), input_shape=ishape, data_format=cf, activation='relu'))
        self.model.add(Conv2D(16, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='linear'))

        return self.model


class RegressorV3:

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        cf = 'channels_first'
        ishape = (1, self.img_rows, self.img_cols)

        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=ishape, data_format=cf, activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='linear'))

        return self.model


class ResNetF:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd
        self.outcomes = outcomes

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        """
        Total params: 1,100,369
        Trainable params: 1,097,913
        Non-trainable params: 2,456     
        """

        input_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetFSE:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
        self.wd = wd

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

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
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            filters = init._keras_shape[channel_axis]
            se_shape = (1, 1, filters)

            se = GlobalAveragePooling2D()(init)
            se = Reshape(se_shape)(se)
            se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)

            x = multiply([init, se])
            return x

        """
        Total params: 1,109,585
        Trainable params: 1,106,673
        Non-trainable params: 2,912  
        """

        input_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetFSEFixed:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
        self.wd = wd

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

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
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            filters = init._keras_shape[channel_axis]
            se_shape = (1, 1, filters)

            se = GlobalAveragePooling2D()(init)
            se = Reshape(se_shape)(se)
            se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)

            x = multiply([init, se])
            return x

        """
        Total params: 1,109,585
        Trainable params: 1,106,673
        Non-trainable params: 2,912  
        """

        input_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=16)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=16)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=16)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=32)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=32)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=32)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=64)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=64)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=64)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=128)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=128)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # squeeze and excite block
        x = squeeze_excite_block(x, ratio=128)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetH:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd
        self.outcomes = outcomes

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        """
        Total params: 3,153,849
        Trainable params: 3,150,485
        Non-trainable params: 3,364  
        """

        input_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetHSE:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd
        self.outcomes = outcomes

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        def squeeze_excite_block(input, ratio=8):
            ''' Create a channel-wise squeeze-excite block
            Args:
                input: input tensor
                filters: number of output filters
            Returns: a keras tensor
            References
            -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
            '''
            init = input
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            filters = init._keras_shape[channel_axis]
            se_shape = (1, 1, filters)

            se = GlobalAveragePooling2D()(init)
            se = Reshape(se_shape)(se)
            se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)

            x = multiply([init, se])
            return x

        """
        Total params: 3,153,849
        Trainable params: 3,150,485
        Non-trainable params: 3,364  
        """

        input_shape = (self.img_rows, self.img_cols, 2)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = squeeze_excite_block(x)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetXt:
    """
    Clean and simple Keras implementation of network architectures described in:
        - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
        - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

    Python 3.
    """

    #
    # image dimensions
    #

    def __init__(self, outcomes, channels, img_rows, img_cols):

        self.outcomes = outcomes
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols

        #
        # network params
        #

        # self.cardinality = 3

    def get_model(self, cardinality=1):

        def residual_network(x):
            """
            ResNeXt by default. For ResNet set `cardinality` = 1 above.

            """

            def add_common_layers(y):
                y = layers.BatchNormalization()(y)
                y = layers.LeakyReLU()(y)

                return y

            def grouped_convolution(y, nb_channels, _strides):
                # when `cardinality` == 1 this is just a standard convolution
                if cardinality == 1:
                    return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,
                                         data_format='channels_first', padding='same')(y)
                # print('cardinality:', cardinality)
                # print('nb_channels:', nb_channels)
                assert not nb_channels % cardinality
                _d = nb_channels // cardinality

                # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
                # and convolutions are separately performed within each group
                groups = []
                for j in range(cardinality):
                    group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                    groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, data_format='channels_first',
                                                padding='same')(group))

                # the grouped convolutional layer concatenates them as the outputs of the layer
                y = layers.concatenate(groups)

                return y

            def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
                """
                Our network consists of a stack of residual blocks. These blocks have the same topology,
                and are subject to two simple rules:

                - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
                - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
                """
                shortcut = y

                # we modify the residual building block as a bottleneck design to make the network more economical
                y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), data_format='channels_first',
                                  padding='same')(y)
                y = add_common_layers(y)

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                y = grouped_convolution(y, nb_channels_in, _strides=_strides)
                y = add_common_layers(y)

                y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), data_format='channels_first',
                                  padding='same')(y)
                # batch normalization is employed after aggregating the transformations and before adding to the shortcut
                y = layers.BatchNormalization()(y)

                # identity shortcuts used directly when the input and output are of the same dimensions
                if _project_shortcut or _strides != (1, 1):
                    # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                    # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                    shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides,
                                             data_format='channels_first', padding='same')(shortcut)
                    shortcut = layers.BatchNormalization()(shortcut)

                y = layers.add([shortcut, y])

                # relu is performed right after each batch normalization,
                # expect for the output of the block where relu is performed after the adding to the shortcut
                y = layers.LeakyReLU()(y)

                return y

            # conv1
            x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', data_format='channels_first')(x)
            x = add_common_layers(x)

            # conv2
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(x)
            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

            # conv3
            for i in range(4):
                # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 256, 512, _strides=strides)

            # conv4
            for i in range(6):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 512, 1024, _strides=strides)

            # conv5
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 1024, 2048, _strides=strides)

            x = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
            x = layers.Dense(self.outcomes, activation='linear')(x)

            return x

        image_tensor = layers.Input(shape=(self.channels, self.img_rows, self.img_cols))
        network_output = residual_network(image_tensor)

        model = models.Model(inputs=[image_tensor], outputs=[network_output])

        return model


class ResNetI:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd
        self.outcomes = outcomes

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        """
        Total params: 2,050,721
        Trainable params: 2,047,921
        Non-trainable params: 2,800
        """

        input_shape = (self.channels, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=32, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=4)(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class DenseNet:

    def __init__(self, channels, img_rows, img_cols, outcomes, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                 nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                 subsample_initial_block=False):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
        self.depth = depth
        self.nb_dense_block = nb_dense_block
        self.growth_rate = growth_rate
        self.nb_filter = nb_filter
        self.nb_layers_per_block = nb_layers_per_block
        self.bottleneck = bottleneck
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.subsample_initial_block = subsample_initial_block

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        model = densenetlst.DenseNet(input_shape=input_shape,
                                     depth=self.depth,
                                     nb_dense_block=self.nb_dense_block,
                                     growth_rate=self.growth_rate,
                                     nb_filter=self.nb_filter,
                                     nb_layers_per_block=self.nb_layers_per_block,
                                     bottleneck=self.bottleneck,
                                     reduction=self.reduction,
                                     dropout_rate=self.dropout_rate,
                                     subsample_initial_block=self.subsample_initial_block,
                                     weight_decay=self.weight_decay,
                                     classes=self.outcomes,
                                     activation='linear')

        return model


class BaseLine: # AAA when you select VGG16N, you call THIS ONE!

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.outcomes = outcomes
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(self.img_rows, self.img_cols, self.channels)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.outcomes, activation='linear'))

        return self.model


class VGG16N: # AAA you are not calling this one, when you select VGG16N, but BaseLine!!!

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.outcomes = outcomes
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):

        input_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=input_shape)

        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Dense(4096, name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layers.Dense(self.outcomes, activation='linear')(x)

        # Create model.
        self.model = models.Model(inputs, x, name='vgg16N')

        return self.model


class VGG16:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='regression', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="vgg16")

        return model


class VGG19:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='regression', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="vgg19")

        return model


'''
class ResNeXt:

    def __init__(self, outcomes, channels, img_rows, img_cols, depth, cardinality, width, weight_decay):
        self.outcomes = outcomes
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.depth = depth
        self.cardinality = cardinality
        self.width = width
        self.weight_decay = weight_decay

    def get_model(self):
        model = resnext.ResNext((self.img_rows, self.img_cols, self.channels),
                                self.depth,
                                self.cardinality,
                                self.width,
                                self.weight_decay,
                                classes=self.outcomes,
                                activation='linear')

        return model
'''