import warnings
warnings.simplefilter('ignore'
import argparse
import datetime
import multiprocessing as mp
import random
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from classifiers import DenseNet
from classifiers import ResNetB
from clr import LRFinder
from generators import DataGeneratorC
from utils import get_all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.', required=True)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)
    parser.add_argument(
        '--phase', type=str, default='', help='Process phase.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    # avoid validation deadlock problem
    mp.set_start_method('spawn', force=True)

    # cmd line parameters
    folders = FLAGS.dirs
    workers = FLAGS.workers
    phase = FLAGS.phase

    # hard coded parameters
    batch_size = 64
    wd = 1e-7  # weight decay

    img_rows, img_cols = 100, 100
    channels = 2

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    n_files = len(h5files)
    val_per = 0.2
    tv_idx = int(n_files * (1 - val_per))
    training_files = h5files[:tv_idx]
    validation_files = h5files[tv_idx:]

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorC(training_files, batch_size=batch_size, arrival_time=True, shuffle=True)

    print('Building validation generator...')
    validation_generator = DataGeneratorC(validation_files, batch_size=batch_size, arrival_time=True, shuffle=True)

    print('Getting validation data...')
    steps_done = 0
    steps = int(len(validation_generator)*0.5)      # take half of the available validation data

    enqueuer = OrderedEnqueuer(validation_generator, use_multiprocessing=True)
    enqueuer.start(workers=workers, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    X_val = []
    Y_val = []

    while steps_done < steps:
        generator_output = next(output_generator)
        x, y = generator_output

        X_val.append(x)
        Y_val.append(y)

        steps_done += 1
        progbar.update(steps_done)

    X_val = np.array(X_val).reshape((steps*batch_size, img_rows, img_cols, channels))
    Y_val = np.array(Y_val).reshape((steps*batch_size))

    print('XVal shapes:', X_val.shape)
    print('YVal shapes:', Y_val.shape)

    if phase == 'lr':
        # lr finder
        model_name = 'DenseNet'
        depth = 64
        growth_rate = 12
        bottleneck = True
        reduction = 0.5
        densenet = DenseNet(channels, img_rows, img_cols, depth=depth, growth_rate=growth_rate, bottleneck=bottleneck,
                            reduction=reduction, weight_decay=0)
        model = densenet.get_model()

        # create a folder to keep model & results
        now = datetime.datetime.now()
        root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
        mkdir(root_dir)

        sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

        lrf = LRFinder(num_samples=len(training_generator) * batch_size - 1,
                       batch_size=batch_size,
                       minimum_lr=1e-6,
                       maximum_lr=1,
                       validation_data=(X_val, Y_val),
                       lr_scale='exp',
                       save_dir=root_dir + '/weights/')

        callbacks = [lrf]

        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit_generator(generator=training_generator,
                            validation_data=(X_val, Y_val),
                            epochs=1,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)

        # plot the previous values if present

        losses, lrs = LRFinder.restore_schedule_from_dir(root_dir + '/weights/',
                                                         clip_beginning=10,
                                                         clip_endding=5)

        plt.plot(lrs, losses)
        plt.title('Learning rate vs Loss')
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.savefig(root_dir + '/lr.png')

        # plt.show()

    if phase == 'momentum':

        model_dir = '/home/nmarinel/ctasoft/cta-lstchain/cnn/DenseNet_2019-04-02_19-56'  # <---------set model dir here
        max_lr = 3.16e-4
        min_lr = max_lr / 10

        MOMENTUMS = [0.8, 0.9, 0.95, 0.99]

        for momentum in MOMENTUMS:
            print('MOMENTUM:', momentum)

            K.clear_session()

            # lr finder
            model_name = 'DenseNet'
            depth = 64
            growth_rate = 12
            bottleneck = True
            reduction = 0.5
            densenet = DenseNet(channels, img_rows, img_cols, depth=depth, growth_rate=growth_rate,
                                bottleneck=bottleneck,
                                reduction=reduction, weight_decay=0)
            model = densenet.get_model()

            # lr finder
            lrf = LRFinder(num_samples=(int(len(training_generator)*0.1) + 1) * batch_size,
                           # use this one to use the entire dataset
                           # num_samples=(len(training_generator) + 1) * batch_size,
                           batch_size=batch_size,
                           minimum_lr=min_lr,
                           maximum_lr=max_lr,
                           validation_data=(X_val, Y_val),
                           lr_scale='linear',
                           save_dir=model_dir + '/momentum/momentum-%s/' % str(
                               momentum))

            sgd = optimizers.SGD(lr=max_lr, momentum=momentum, nesterov=True)

            callbacks = [lrf]

            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

            model.fit_generator(generator=training_generator,
                                # it can be commented if you want to use the entire dataset
                                steps_per_epoch=int(len(training_generator)*0.1),
                                validation_data=(X_val, Y_val),
                                epochs=1,
                                verbose=1,
                                use_multiprocessing=True,
                                workers=FLAGS.workers,
                                shuffle=False,
                                callbacks=callbacks)

        for momentum in MOMENTUMS:
            directory = model_dir + '/momentum/momentum-%s/' % str(momentum)

            losses, lrs = LRFinder.restore_schedule_from_dir(directory, 10, 5)
            plt.plot(lrs, losses, label='momentum=%0.2f' % momentum)

        plt.title("Momentum")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.savefig(model_dir + '/momentum.png')
        # plt.show()

    if phase == 'wd':

        model_dir = '/home/nmarinel/ctasoft/cta-lstchain/cnn/DenseNet_2019-04-02_19-56'  # <---------set model dir here
        max_lr = 3.16e-4
        min_lr = max_lr / 10
        momentum = 0.99

        # INITIAL WEIGHT DECAY FACTORS
        # WEIGHT_DECAY_FACTORS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

        # FINEGRAINED WEIGHT DECAY FACTORS
        WEIGHT_DECAY_FACTORS = [1e-3, 1e-4, 1e-5]

        for weight_decay in WEIGHT_DECAY_FACTORS:
            print('WEIGHT_DECAY:', weight_decay)

            K.clear_session()

            # lr finder
            model_name = 'DenseNet'
            depth = 64
            growth_rate = 12
            bottleneck = True
            reduction = 0.5
            densenet = DenseNet(channels, img_rows, img_cols, depth=depth, growth_rate=growth_rate,
                                bottleneck=bottleneck,
                                reduction=reduction, weight_decay=weight_decay)
            model = densenet.get_model()

            # lr finder
            lrf = LRFinder(num_samples=(int(len(training_generator)*0.1) + 1) * batch_size,
                           # use this one to use the entire dataset
                           # num_samples=(len(training_generator) + 1) * batch_size,
                           batch_size=batch_size,
                           minimum_lr=min_lr,
                           maximum_lr=max_lr,
                           validation_data=(X_val, Y_val),
                           lr_scale='linear',
                           save_dir=model_dir + '/weight_decay/weight_decay-%s/' % str(weight_decay))

            sgd = optimizers.SGD(lr=max_lr, momentum=momentum, nesterov=True)

            callbacks = [lrf]

            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

            model.fit_generator(generator=training_generator,
                                # it can be commented if you want to use the entire dataset
                                steps_per_epoch=int(len(training_generator) * 0.1),
                                validation_data=(X_val, Y_val),
                                epochs=1,
                                verbose=1,
                                use_multiprocessing=True,
                                workers=FLAGS.workers,
                                shuffle=False,
                                callbacks=callbacks)

        for weight_decay in WEIGHT_DECAY_FACTORS:
            directory = model_dir + '/weight_decay/weight_decay-%s/' % str(weight_decay)

            losses, lrs = LRFinder.restore_schedule_from_dir(directory, 10, 5)
            plt.plot(lrs, losses, label='weight_decay=%0.7f' % weight_decay)

        plt.title("Weight Decay")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.savefig(model_dir + '/wd.png')
        # plt.show()
