import warnings
warnings.simplefilter('ignore')

import sys

from regressors import ResNetF, DenseNet, ResNetFSE, BaseLine, \
     ResNet50, VGG16, VGG16N, VGG19, DenseNet121
#, ResNeXt



def regressor_selector(model_name, hype_print, channels, img_rows, img_cols, outcomes):
    if model_name == 'ResNetF':
        wd = 1e-5
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
    elif model_name == 'DenseNet121':
        net = DenseNet121(outcomes, channels, img_rows, img_cols)
        model = net.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNet50':
        resnet = ResNet50(outcomes, channels, img_rows, img_cols)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetFSE':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetFSE(outcomes, channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'BaseLine':
        bl = BaseLine(outcomes, channels, img_rows, img_cols)
        model = bl.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'VGG16':
        vgg16 = VGG16(outcomes, channels, img_rows, img_cols)
        model = vgg16.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'VGG16N':
        vgg16N = VGG16N(outcomes, channels, img_rows, img_cols)
        model = vgg16N.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'VGG19':
        vgg19 = VGG19(outcomes, channels, img_rows, img_cols)
        model = vgg19.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    return model, hype_print