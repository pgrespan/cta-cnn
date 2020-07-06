import warnings
warnings.simplefilter('ignore')

import sys
from lst_cnns import LST_VGG16, LST_ResNet50#, LST_DenseNet
import regressors
from regressors import ResNetF, DenseNet, ResNetFSE, BaseLine, VGG16N, \
    Xception, InceptionV3, InceptionResNetV2, \
    ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, \
    VGG16, VGG19, DenseNet121
#, ResNeXt


def regressor_selector(model_name, hype_print, channels, img_rows, img_cols, outcomes, feature):

    if model_name == 'ResNetF':
        wd = 0. #1e-5
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetF(outcomes, channels, img_rows, img_cols, wd)
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
                            outcomes,
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
    elif model_name == 'DenseNet169N':
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
                            outcomes,
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
    elif model_name == 'LST_VGG16':
        dropout_rate = 0.5
        wd = 1e-5
        net = LST_VGG16(
            channels=channels,
            img_rows=img_rows,
            img_cols=img_cols,
            outcomes=outcomes,
            dropout_rate=dropout_rate,
            weight_decay=wd
        )
        model = net.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        hype_print += '\n' + 'weight_decay: ' + str(wd)
    #elif model_name == 'LST_DenseNet':
    #    net = LST_DenseNet(outcomes, channels, img_rows, img_cols)
    #    model = net.get_model()
    #    params = model.count_params()
    #    hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'LST_ResNet50':
        dropout_rate = 0.0
        wd = 0.0
        net = LST_ResNet50(
            channels=channels,
            img_rows=img_rows,
            img_cols=img_cols,
            outcomes=outcomes,
            dropout_rate=dropout_rate,
            weight_decay=wd
        )
        model = net.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        #hype_print += '\n' + 'dropout_rate: ' + str(dropout_rate)
        #hype_print += '\n' + 'weight_decay: ' + str(wd)
    else:
        net = getattr(regressors, model_name)
        net = net(outcomes, channels, img_rows, img_cols)
        model = net.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)

    return model, hype_print