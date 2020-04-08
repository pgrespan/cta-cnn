import sys

from keras.layers import Dense
from keras.utils.data_utils import get_file

from classifiers import ClassifierV1, ClassifierV2, ClassifierV3, ResNet, ResNetA, ResNetB, ResNetC, ResNetD, ResNetE, \
    ResNetF, ResNetG, ResNetH, DenseNet, ResNetFSE, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, NASNetLarge, \
    NASNetA, ResNetFSEA, BaseLine, ResNeXt, VGG16, ResNetFSEFixed, DenseNetSE

DENSENET_169_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32-no-top.h5'


def select_classifier(model_name, hype_print, channels, img_rows, img_cols):
    if model_name == 'ClassifierV1':
        class_v1 = ClassifierV1(channels, img_rows, img_cols)
        model = class_v1.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ClassifierV2':
        class_v2 = ClassifierV2(img_rows, img_cols)
        model = class_v2.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ClassifierV3':
        class_v3 = ClassifierV3(channels, img_rows, img_cols)
        model = class_v3.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet20V1':
        resnet = ResNet(img_rows, img_cols)
        model = resnet.get_model(1, 3)
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet32V1':
        resnet = ResNet(img_rows, img_cols)
        model = resnet.get_model(1, 5)
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetA':
        resnet = ResNetA(img_rows, img_cols)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetB':
        wd = 3e-6
        print('Weight decay: ', wd)
        resnet = ResNetB(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetC':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetC(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetD':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetD(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetE':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetE(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetF':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetF(channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetG':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetG(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetH':
        wd = 1e-3
        print('Weight decay: ', wd)
        resnet = ResNetH(img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'DenseNet':
        depth = 64
        growth_rate = 12
        bottleneck = True
        reduction = 0.5
        # subsample_initial_block = True
        # wd = 1e-5
        # dropout_rate = 0
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            growth_rate=growth_rate,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            # dropout_rate=dropout_rate,
                            # subsample_initial_block=subsample_initial_block,
                            )
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
        # hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        # hype_print += '\n' + 'subsample_initial_block: ' + str(subsample_initial_block)
        # hype_print += '\n' + 'Weight decay: ' + str(wd)
    elif model_name == 'DenseNet169ImageNet':
        depth = 169
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers_per_block = [6, 12, 32, 32]
        bottleneck = True
        reduction = 0.5
        dropout_rate = 0.0
        subsample_initial_block = True
        weights_path = get_file('DenseNet-BC-169-32-no-top.h5',
                                DENSENET_169_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='89c19e8276cfd10585d5fadc1df6859e')
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate,
                            nb_filter=nb_filter,
                            nb_layers_per_block=nb_layers_per_block,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            dropout_rate=dropout_rate,
                            subsample_initial_block=subsample_initial_block,
                            include_top=False)
        model = densenet.get_model()
        model.load_weights(weights_path)
        model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))
        params = model.count_params()
        hype_print += '\n' + 'DenseNet169 with transfer learning from ImageNet'
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'nb_dense_block: ' + str(nb_dense_block)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'nb_filter: ' + str(nb_filter)
        hype_print += '\n' + 'nb_layers_per_block: ' + str(nb_layers_per_block)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
        hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        hype_print += '\n' + 'subsample_initial_block: ' + str(subsample_initial_block)
    elif model_name == 'DenseNet169':
        depth = 169
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers_per_block = [6, 12, 32, 32]
        bottleneck = True
        reduction = 0.5
        dropout_rate = 0.25
        subsample_initial_block = True
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate,
                            nb_filter=nb_filter,
                            nb_layers_per_block=nb_layers_per_block,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            dropout_rate=dropout_rate,
                            subsample_initial_block=subsample_initial_block)
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'nb_dense_block: ' + str(nb_dense_block)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'nb_filter: ' + str(nb_filter)
        hype_print += '\n' + 'nb_layers_per_block: ' + str(nb_layers_per_block)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
        hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        hype_print += '\n' + 'subsample_initial_block: ' + str(subsample_initial_block)
    elif model_name == 'DenseNetA':
        nb_dense_block = 3
        growth_rate = 16
        nb_filter = 64
        nb_layers_per_block = [8, 6, 16]
        bottleneck = True
        reduction = 0.5
        dropout_rate = 0.0
        subsample_initial_block = False
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate,
                            nb_filter=nb_filter,
                            nb_layers_per_block=nb_layers_per_block,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            dropout_rate=dropout_rate,
                            subsample_initial_block=subsample_initial_block)
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'nb_dense_block: ' + str(nb_dense_block)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'nb_filter: ' + str(nb_filter)
        hype_print += '\n' + 'nb_layers_per_block: ' + str(nb_layers_per_block)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
        hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        hype_print += '\n' + 'subsample_initial_block: ' + str(subsample_initial_block)
    elif model_name == 'DenseNetB':
        depth = 79
        growth_rate = 12
        bottleneck = True
        reduction = 0.5
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            growth_rate=growth_rate,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            )
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
    elif model_name == 'DenseNetC':
        depth = 94
        growth_rate = 12
        bottleneck = False
        reduction = 0
        subsample_initial_block = True
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            growth_rate=growth_rate,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            subsample_initial_block=subsample_initial_block
                            )
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
    elif model_name == 'DenseNetD':
        depth = 34
        growth_rate = 32
        bottleneck = True
        reduction = 0.3
        subsample_initial_block = True
        densenet = DenseNet(channels,
                            img_rows,
                            img_cols,
                            depth=depth,
                            growth_rate=growth_rate,
                            bottleneck=bottleneck,
                            reduction=reduction,
                            subsample_initial_block=subsample_initial_block
                            )
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
    elif model_name == 'ResNetFSE':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetFSE(channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetFSEFixed':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetFSEFixed(channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetFSEA':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetFSEA(channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet18':
        dropout = 0.5
        hype_print += '\n' + 'dropout: ' + str(dropout)
        resnet = ResNet18(channels, img_rows, img_cols, dropout)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet34':
        dropout = 0.5
        hype_print += '\n' + 'dropout: ' + str(dropout)
        resnet = ResNet34(channels, img_rows, img_cols, dropout)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet50':
        dropout = 0.5
        hype_print += '\n' + 'dropout: ' + str(dropout)
        resnet = ResNet50(channels, img_rows, img_cols, dropout)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet101':
        dropout = 0.5
        hype_print += '\n' + 'dropout: ' + str(dropout)
        resnet = ResNet101(channels, img_rows, img_cols, dropout)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet152':
        dropout = 0.5
        hype_print += '\n' + 'dropout: ' + str(dropout)
        resnet = ResNet152(channels, img_rows, img_cols, dropout)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'NASNetLarge':
        resnet = NASNetLarge(channels, img_rows, img_cols)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'NASNetA':
        resnet = NASNetA(channels, img_rows, img_cols)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'BaseLine':
        bl = BaseLine(channels, img_rows, img_cols)
        model = bl.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'VGG16':
        vgg = VGG16(channels, img_rows, img_cols)
        model = vgg.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNeXt56-8-64':
        depth = 56
        cardinality = 8
        width = 64
        weight_decay = 5e-4
        rxt = ResNeXt(channels, img_rows, img_cols, depth, cardinality, width, weight_decay)
        model = rxt.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNeXt29-4-8':
        depth = 29
        cardinality = 4
        width = 8
        weight_decay = 5e-4
        rxt = ResNeXt(channels, img_rows, img_cols, depth, cardinality, width, weight_decay)
        model = rxt.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'DenseNetSE':
        depth = 64
        growth_rate = 12
        bottleneck = True
        reduction = 0.5
        # subsample_initial_block = True
        # wd = 1e-5
        # dropout_rate = 0
        densenet = DenseNetSE(channels,
                              img_rows,
                              img_cols,
                              depth=depth,
                              growth_rate=growth_rate,
                              bottleneck=bottleneck,
                              reduction=reduction,
                              # dropout_rate=dropout_rate,
                              # subsample_initial_block=subsample_initial_block,
                              )
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
        # hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        # hype_print += '\n' + 'subsample_initial_block: ' + str(subsample_initial_block)
        # hype_print += '\n' + 'Weight decay: ' + str(wd)
    else:
        print('Model name not valid')
        sys.exit(1)

    return model, hype_print
