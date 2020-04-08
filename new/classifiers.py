import keras
import keras_contrib.applications
from keras import backend as K
from keras import layers
from keras import models
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, \
    Activation, Input, Reshape, multiply, GlobalAveragePooling2D, Permute
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2

import densenetlst
# import resnext
# import sedensenetlst


class ClassifierV1:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                              input_shape=(self.img_rows, self.img_cols, self.channels)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1, activation='sigmoid'))

        return self.model


# TODO: batch normalization?


class ClassifierV2:

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        self.model.add(
            Conv2D(16, kernel_size=(3, 3), input_shape=(1, self.img_rows, self.img_cols), data_format='channels_first',
                   activation='relu'))
        self.model.add(Conv2D(16, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model


class ClassifierV3:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        self.model.add(Conv2D(32, (3, 3), input_shape=(self.img_rows, self.img_cols, self.channels), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.20))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.20))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model


class ResNet:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self, version, n):

        def get_resnetv1(input_shape, depth):

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
                              kernel_regularizer=l2(1e-4),
                              data_format="channels_first")

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

            def resnet_v1(input_shape, depth):
                """ResNet Version 1 Model builder [a]
                Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
                Last ReLU is after the shortcut connection.
                At the beginning of each stage, the feature map size is halved (downsampled)
                by a convolutional layer with strides=2, while the number of filters is
                doubled. Within each stage, the layers have the same number filters and the
                same number of filters.
                Features maps sizes:
                stage 0: 32x32, 16
                stage 1: 16x16, 32
                stage 2:  8x8,  64
                The Number of parameters is approx the same as Table 6 of [a]:
                ResNet20 0.27M
                ResNet32 0.46M
                ResNet44 0.66M
                ResNet56 0.85M
                ResNet110 1.7M
                # Arguments
                    input_shape (tensor): shape of input image tensor
                    depth (int): number of core convolutional layers
                    num_classes (int): number of classes (CIFAR10 has 10)
                # Returns
                    model (Model): Keras model instance
                """
                if (depth - 2) % 6 != 0:
                    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
                # Start model definition.
                num_filters = 16
                num_res_blocks = int((depth - 2) / 6)

                inputs = Input(shape=input_shape)
                x = resnet_layer(inputs=inputs)
                # Instantiate the stack of residual units
                for stack in range(3):
                    for res_block in range(num_res_blocks):
                        strides = 1
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            strides = 2  # downsample
                        y = resnet_layer(inputs=x,
                                         num_filters=num_filters,
                                         strides=strides)
                        y = resnet_layer(inputs=y,
                                         num_filters=num_filters,
                                         activation=None)
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            # linear projection residual shortcut connection to match
                            # changed dims
                            x = resnet_layer(inputs=x,
                                             num_filters=num_filters,
                                             kernel_size=1,
                                             strides=strides,
                                             activation=None,
                                             batch_normalization=False)
                        x = keras.layers.add([x, y])
                        x = Activation('relu')(x)
                    num_filters *= 2

                # Add classifier on top.
                # v1 does not use BN after last shortcut connection-ReLU
                x = AveragePooling2D(pool_size=8, data_format='channels_first')(x)
                y = Flatten()(x)
                outputs = Dense(1,
                                activation='sigmoid',
                                kernel_initializer='he_normal')(y)

                # Instantiate model.
                model = Model(inputs=inputs, outputs=outputs)
                return model

            return resnet_v1(input_shape, depth)

        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

        # Computed depth from supplied model parameter n
        if version == 1:
            depth = n * 6 + 2

            # Model name, depth and version
            model_type = 'ResNet%dv%d' % (depth, version)

            print(model_type)

            model = get_resnetv1((1, self.img_rows, self.img_cols), depth)

        # elif version == 2:
        #    depth = n * 9 + 2

        return model


class ResNetA:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self):

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
                          kernel_regularizer=l2(1e-4),
                          data_format="channels_first")

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

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
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
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=8, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetB:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (8, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetC:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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
        
        batch_size 128
        adam
        
        Total params: 23,161
        Trainable params: 22,261
        Non-trainable params: 900
        __________________________________________________________________________________________________
        Epoch 1/75
        6564/6564 [==============================] - 1514s 231ms/step - loss: 0.5753 - acc: 0.6771 - val_loss: 0.5251 - val_acc: 0.7201
        Epoch 2/75
        6564/6564 [==============================] - 1487s 226ms/step - loss: 0.5078 - acc: 0.7370 - val_loss: 0.5305 - val_acc: 0.7180
        Epoch 3/75
        6564/6564 [==============================] - 1494s 228ms/step - loss: 0.4926 - acc: 0.7469 - val_loss: 0.5266 - val_acc: 0.7200
        Epoch 4/75
        6564/6564 [==============================] - 1407s 214ms/step - loss: 0.4843 - acc: 0.7525 - val_loss: 0.5110 - val_acc: 0.7357
        Epoch 5/75
        6564/6564 [==============================] - 1475s 225ms/step - loss: 0.4782 - acc: 0.7558 - val_loss: 0.5173 - val_acc: 0.7213
        Epoch 6/75
        6564/6564 [==============================] - 1503s 229ms/step - loss: 0.4739 - acc: 0.7585 - val_loss: 0.4838 - val_acc: 0.7515
        Epoch 7/75
        6564/6564 [==============================] - 1484s 226ms/step - loss: 0.4702 - acc: 0.7609 - val_loss: 0.5082 - val_acc: 0.7339
        Epoch 8/75
        6564/6564 [==============================] - 1484s 226ms/step - loss: 0.4672 - acc: 0.7624 - val_loss: 0.4913 - val_acc: 0.7451
        Epoch 9/75
        6564/6564 [==============================] - 1493s 227ms/step - loss: 0.4647 - acc: 0.7643 - val_loss: 0.5306 - val_acc: 0.7184
        Epoch 10/75
        6564/6564 [==============================] - 1496s 228ms/step - loss: 0.4628 - acc: 0.7653 - val_loss: 0.4680 - val_acc: 0.7612
        Epoch 11/75
        6564/6564 [==============================] - 1496s 228ms/step - loss: 0.4614 - acc: 0.7653 - val_loss: 0.4946 - val_acc: 0.7380
        Epoch 12/75
        6564/6564 [==============================] - 1501s 229ms/step - loss: 0.4597 - acc: 0.7668 - val_loss: 0.4843 - val_acc: 0.7482
        Epoch 13/75
        6564/6564 [==============================] - 1500s 229ms/step - loss: 0.4580 - acc: 0.7674 - val_loss: 0.5041 - val_acc: 0.7302
        Epoch 14/75
        4774/6564 [====================>.........] - ETA: 6:21 - loss: 0.4570 - acc: 0.7684
        
        """

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetD:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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
        batch size 128
        adam
        overfits
        
        Total params: 181,057
        Trainable params: 179,457
        Non-trainable params: 1,600
        __________________________________________________________________________________________________
        Epoch 1/75
        6607/6607 [==============================] - 1571s 238ms/step - loss: 0.5509 - acc: 0.6958 - val_loss: 0.5315 - val_acc: 0.7138
        Epoch 2/75
        6607/6607 [==============================] - 1559s 236ms/step - loss: 0.4804 - acc: 0.7533 - val_loss: 0.5282 - val_acc: 0.7161
        Epoch 3/75
        6607/6607 [==============================] - 1598s 242ms/step - loss: 0.4653 - acc: 0.7621 - val_loss: 0.4703 - val_acc: 0.7608
        Epoch 4/75
        6607/6607 [==============================] - 1584s 240ms/step - loss: 0.4554 - acc: 0.7675 - val_loss: 0.4998 - val_acc: 0.7352
        Epoch 5/75
        6607/6607 [==============================] - 1587s 240ms/step - loss: 0.4482 - acc: 0.7720 - val_loss: 0.4786 - val_acc: 0.7558
        Epoch 6/75
        6607/6607 [==============================] - 1572s 238ms/step - loss: 0.4433 - acc: 0.7754 - val_loss: 0.4537 - val_acc: 0.7678
        Epoch 7/75
        6607/6607 [==============================] - 1628s 246ms/step - loss: 0.4384 - acc: 0.7776 - val_loss: 0.4533 - val_acc: 0.7710
        Epoch 8/75
        6607/6607 [==============================] - 1629s 247ms/step - loss: 0.4347 - acc: 0.7796 - val_loss: 0.4749 - val_acc: 0.7549
        Epoch 9/75
        6607/6607 [==============================] - 1593s 241ms/step - loss: 0.4308 - acc: 0.7820 - val_loss: 0.4525 - val_acc: 0.7712
        Epoch 10/75
        6607/6607 [==============================] - 1579s 239ms/step - loss: 0.4274 - acc: 0.7838 - val_loss: 0.4478 - val_acc: 0.7739
        Epoch 11/75
        6607/6607 [==============================] - 1564s 237ms/step - loss: 0.4242 - acc: 0.7856 - val_loss: 0.4614 - val_acc: 0.7653
        Epoch 12/75
        6607/6607 [==============================] - 1578s 239ms/step - loss: 0.4208 - acc: 0.7871 - val_loss: 0.4569 - val_acc: 0.7670
        Epoch 13/75
        6607/6607 [==============================] - 1569s 237ms/step - loss: 0.4176 - acc: 0.7890 - val_loss: 0.4648 - val_acc: 0.7623
        Epoch 14/75
        6607/6607 [==============================] - 1583s 240ms/step - loss: 0.4142 - acc: 0.7911 - val_loss: 0.4524 - val_acc: 0.7685
        Epoch 15/75
        6607/6607 [==============================] - 1599s 242ms/step - loss: 0.4109 - acc: 0.7926 - val_loss: 0.4541 - val_acc: 0.7684
        Epoch 16/75
        6607/6607 [==============================] - 1594s 241ms/step - loss: 0.4077 - acc: 0.7940 - val_loss: 0.4498 - val_acc: 0.7719
        Epoch 17/75
        6607/6607 [==============================] - 1581s 239ms/step - loss: 0.4043 - acc: 0.7961 - val_loss: 0.4516 - val_acc: 0.7714
        Epoch 18/75
        6607/6607 [==============================] - 1613s 244ms/step - loss: 0.4013 - acc: 0.7980 - val_loss: 0.4518 - val_acc: 0.7707
        Epoch 19/75
        6607/6607 [==============================] - 1591s 241ms/step - loss: 0.3982 - acc: 0.7996 - val_loss: 0.4596 - val_acc: 0.7669
        Epoch 20/75
        6607/6607 [==============================] - 1557s 236ms/step - loss: 0.3947 - acc: 0.8016 - val_loss: 0.4650 - val_acc: 0.7637
        Epoch 21/75
        6607/6607 [==============================] - 1579s 239ms/step - loss: 0.3914 - acc: 0.8032 - val_loss: 0.4634 - val_acc: 0.7651
        Epoch 22/75
        6607/6607 [==============================] - 1593s 241ms/step - loss: 0.3881 - acc: 0.8051 - val_loss: 0.4672 - val_acc: 0.7634
        Epoch 23/75
        6607/6607 [==============================] - 1596s 242ms/step - loss: 0.3847 - acc: 0.8071 - val_loss: 0.4686 - val_acc: 0.7628
        Epoch 24/75
        6607/6607 [==============================] - 1573s 238ms/step - loss: 0.3820 - acc: 0.8085 - val_loss: 0.4717 - val_acc: 0.7628
        Epoch 25/75
        6607/6607 [==============================] - 1575s 238ms/step - loss: 0.3788 - acc: 0.8104 - val_loss: 0.4795 - val_acc: 0.7545
        Epoch 26/75
        6607/6607 [==============================] - 1581s 239ms/step - loss: 0.3754 - acc: 0.8122 - val_loss: 0.4718 - val_acc: 0.7619
        Epoch 27/75
        2085/6607 [========>.....................] - ETA: 15:18 - loss: 0.3672 - acc: 0.8173
        
        

        """

        input_shape = (1, self.img_rows, self.img_cols)

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
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetE:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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
        Total params: 179,441
        Trainable params: 177,737
        Non-trainable params: 1,704
        """

        input_shape = (1, self.img_rows, self.img_cols)

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

        # stack 1
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

        # stack 2
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

        # stack 3
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

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(y)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        # outputs = Dense(1, activation='sigmoid')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetF:

    def __init__(self, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetG:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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
        Total params: 1,783,865
        Trainable params: 1,781,005
        Non-trainable params: 2,860    
        """

        input_shape = (1, self.img_rows, self.img_cols)

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

        x = AveragePooling2D(pool_size=2, data_format='channels_first')(y)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetH:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
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
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

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

        input_shape = (1, self.img_rows, self.img_cols)

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

        x = AveragePooling2D(pool_size=2, data_format='channels_first')(y)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class DenseNet:

    def __init__(self, channels, img_rows, img_cols, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                 nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                 subsample_initial_block=False, include_top=True):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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
        self.include_top = include_top

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
                                     classes=1,
                                     activation='sigmoid',
                                     include_top=self.include_top)

        return model


class ResNetFSE:

    def __init__(self, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetFSEFixed:

    def __init__(self, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetFSEA:

    def __init__(self, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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

        x = AveragePooling2D(pool_size=2)(y)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNet18:

    def __init__(self, channels, img_rows, img_cols, dropout):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dropout = dropout

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        # ResNet50
        model = keras_contrib.applications.resnet.ResNet(input_shape=input_shape,
                                                         block='basic',
                                                         dropout=self.dropout,
                                                         repetitions=[2, 2, 2, 2],
                                                         residual_unit='v1',
                                                         classes=1,
                                                         activation='sigmoid')

        return model


class ResNet34:

    def __init__(self, channels, img_rows, img_cols, dropout):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dropout = dropout

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        # ResNet50
        model = keras_contrib.applications.resnet.ResNet(input_shape=input_shape,
                                                         block='basic',
                                                         dropout=self.dropout,
                                                         repetitions=[3, 4, 6, 3],
                                                         residual_unit='v1',
                                                         classes=1,
                                                         activation='sigmoid')

        return model


class ResNet50:

    def __init__(self, channels, img_rows, img_cols, dropout):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dropout = dropout

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        # ResNet50
        model = keras_contrib.applications.resnet.ResNet(input_shape=input_shape,
                                                         block='bottleneck',
                                                         dropout=self.dropout,
                                                         repetitions=[3, 4, 6, 3],
                                                         residual_unit='v2',
                                                         classes=1,
                                                         activation='sigmoid')

        return model


class ResNet101:

    def __init__(self, channels, img_rows, img_cols, dropout):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dropout = dropout

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        # ResNet101
        model = keras_contrib.applications.resnet.ResNet(input_shape=input_shape,
                                                         block='bottleneck',
                                                         dropout=self.dropout,
                                                         repetitions=[3, 4, 23, 3],
                                                         residual_unit='v2',
                                                         classes=1,
                                                         activation='sigmoid')

        return model


class ResNet152:

    def __init__(self, channels, img_rows, img_cols, dropout):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dropout = dropout

    def get_model(self):
        input_shape = (self.img_rows, self.img_cols, self.channels)

        # ResNet152
        model = keras_contrib.applications.resnet.ResNet(input_shape=input_shape,
                                                         block='bottleneck',
                                                         dropout=self.dropout,
                                                         repetitions=[3, 8, 36, 3],
                                                         residual_unit='v2',
                                                         classes=1,
                                                         activation='sigmoid')

        return model


class NASNetLarge:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self):
        model = keras_contrib.applications.nasnet.NASNetLarge(input_shape=(self.img_rows, self.img_cols, self.channels),
                                                              classes=1,
                                                              activation='sigmoid',
                                                              include_top=True,
                                                              weights=None)

        return model


class NASNetA:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self):
        model = keras_contrib.applications.nasnet.NASNet(input_shape=(self.img_rows, self.img_cols, self.channels),
                                                         initial_reduction=True,
                                                         classes=1,
                                                         activation='sigmoid',
                                                         include_top=True,
                                                         weights=None)

        return model


class BaseLine:

    def __init__(self, channels, img_rows, img_cols):
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
        self.model.add(Dense(units=1, activation='sigmoid'))

        return self.model


class ResNeXt:

    def __init__(self, channels, img_rows, img_cols, depth, cardinality, width, weight_decay):
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
                                classes=1,
                                activation='sigmoid')

        return model


class VGG16:

    def __init__(self, channels, img_rows, img_cols):
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
        x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
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
        x = layers.Dense(1, activation='sigmoid')(x)

        # Create model.
        self.model = models.Model(inputs, x, name='vgg16')

        return self.model


class DenseNetSE:

    def __init__(self, channels, img_rows, img_cols, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                 nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                 subsample_initial_block=False, include_top=True):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
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
        self.include_top = include_top

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
                                       classes=1,
                                       activation='sigmoid',
                                       include_top=self.include_top)

        return model

