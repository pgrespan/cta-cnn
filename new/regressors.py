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

#########################################################################################################################
#######################################       CNNs FROM KERAS.APPLICATIONS        #######################################
#########################################################################################################################

includetop = False

class Xception:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.Xception(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="InceptionV3")
        return model


class InceptionV3:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.InceptionV3(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="InceptionV3")
        return model


class InceptionResNetV2:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.InceptionResNetV2(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="InceptionResNetV2")
        return model


class NASNetLarge:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.NASNetLarge(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="NASNetLarge")
        return model


class NASNetMobile:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.NASNetMobile(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="NasNetMobile")
        return model


class ResNet50:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet50(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet50")
        return model


class ResNet101:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet101(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet101")
        return model


class ResNet152:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet152(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet152")
        return model


class ResNet50V2:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet50V2(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet50V2")
        return model


class ResNet101V2:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet101V2(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet101V2")
        return model


class ResNet152V2:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.ResNet152V2(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="ResNet152V2")
        return model


class VGG16:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.VGG16(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='regression', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="VGG16")

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
        base = keras.applications.VGG19(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        #x = Flatten(name='flatten')(x)
        x = Dense(1000, activation='linear', name='fc1', kernel_initializer='he_normal')(x)
        x = Dense(self.outcomes, name='regression', activation='linear', kernel_initializer='he_normal')(x)
        model = Model(inputs=input_img, output=x, name="VGG19")

        return model


class DenseNet121:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.densenet.DenseNet121(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        #base = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="DenseNet121")
        return model


class DenseNet169:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.DenseNet169(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="DenseNet169")
        return model


class DenseNet201:

    def __init__(self, outcomes, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)
        input_img = Input(input_shape, name='input_img')
        base = keras.applications.DenseNet201(include_top=includetop, weights=None, input_tensor=input_img, pooling='max')
        x = base.layers[-1].output
        x = Dense(self.outcomes, name='gammaness', activation='linear')(x)
        model = Model(inputs=input_img, output=x, name="DenseNet201")
        return model


#########################################################################################################################
#######################################               CUSTOM CNNS                 #######################################
#########################################################################################################################


'''
class Custom21CL:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.outcomes = outcomes
        self.wd = wd
    def get_model(self):

        def conv_21():

        return model
'''


# Nicola Marinello's CNNs:

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
                         batch_normalization=False,
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
        y = BatchNormalization()(inputs)
        y = resnet_layer(inputs=y, num_filters=16, strides=1)  # output (16, 100, 100)

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
                         batch_normalization=False,
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
        y = BatchNormalization()(inputs)
        y = resnet_layer(inputs=y, num_filters=16, strides=1)  # output (16, 100, 100)

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


