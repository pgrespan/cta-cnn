import warnings
warnings.simplefilter('ignore')

import argparse
import datetime
import multiprocessing as mp
import os
import pickle
import sys
from os import listdir
from os import mkdir
from os.path import isfile, join

import keras
import keras.backend as K
#from adabound import AdaBound
from generators import DataGeneratorR
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
from losseshistory import LossHistoryR
from regressor_selector import regressor_selector
from regressor_test_plots import test_plots
from regressor_tester import tester
from regressor_training_plots import train_plots

from utils import get_all_files
import tensorflow as tf


def regressor_training_main(folders, val_folders, model_name, time, epochs, batch_size, opt, learning_rate, lropf, sd, es,
                            feature, workers, test_dirs, intensity_cut, tb, gpu_fraction, emin, emax):
    ###################################
    # TensorFlow wizardry for GPU dynamic memory allocation
    #if gpu_fraction != 0 and gpu_fraction <= 1:
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a fraction of the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    # remove semaphore warnings
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    # avoid validation deadlock problem
    mp.set_start_method('spawn', force=True)

    # hard coded parameters
    shuffle = True

    img_rows, img_cols = 100, 100
    channels = 1
    if time:
        channels = 2

    # early stopping
    md_es = 0.1  # min delta
    p_es = 50  # patience

    # sgd
    lr = 0.01  # lr
    decay = 1e-4  # decay
    momentum = 0.9  # momentum
    nesterov = True

    # adam
    a_lr = learning_rate
    a_beta_1 = 0.9
    a_beta_2 = 0.999
    a_epsilon = None
    a_decay = 0
    amsgrad = True

    # adabound
    ab_lr = 1e-03
    ab_final_lr = 0.1
    ab_gamma = 1e-03
    ab_weight_decay = 0
    amsbound = False

    # rmsprop
    r_lr = 0.01
    r_rho = 0.9
    r_epsilon = None
    r_decay = 0.0

    # reduce lr on plateau
    f_lrop = 0.1  # factor
    p_lrop = 15  # patience
    md_lrop = 0.005  # min delta
    cd_lrop = 5  # cool down
    mlr_lrop = a_lr / 100  # min lr

    # cuts
    # intensity_cut = 50
    leakage2_intensity_cut = 0.2

    training_files = get_all_files(folders)
    validation_files = get_all_files(val_folders)

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorR(training_files, batch_size=batch_size, arrival_time=time, feature=feature,
                                        shuffle=shuffle, intensity=intensity_cut,
                                        leakage2_intensity=leakage2_intensity_cut)

    if len(val_folders) > 0:
        print('Building validation generator...')
        validation_generator = DataGeneratorR(validation_files, batch_size=batch_size, arrival_time=time,
                                              feature=feature, shuffle=False, intensity=intensity_cut,
                                              leakage2_intensity=leakage2_intensity_cut)

    # class_weight = {0: 1., 1: train_protons/train_gammas}
    # print(class_weight)

    hype_print = '\n' + '======================================HYPERPARAMETERS======================================'

    hype_print += '\n' + 'Image rows: ' + str(img_rows) + ' Image cols: ' + str(img_cols)
    hype_print += '\n' + 'Folders:' + str(folders)
    hype_print += '\n' + 'Model: ' + str(model_name)
    hype_print += '\n' + 'Use arrival time: ' + str(time)
    hype_print += '\n' + 'Epochs:' + str(epochs)
    hype_print += '\n' + 'Batch size: ' + str(batch_size)
    hype_print += '\n' + 'Optimizer: ' + str(opt)
    hype_print += '\n' + 'Feature: ' + str(feature)
    hype_print += '\n' + 'Validation: ' + str(val_folders)
    hype_print += '\n' + 'Test dirs: ' + str(test_dirs)

    hype_print += '\n' + 'intensity_cut: ' + str(intensity_cut)
    hype_print += '\n' + 'leakage2_intensity_cut: ' + str(leakage2_intensity_cut)

    if es:
        hype_print += '\n' + '--- Early stopping ---'
        hype_print += '\n' + 'Min delta: ' + str(md_es)
        hype_print += '\n' + 'Patience: ' + str(p_es)
        hype_print += '\n' + '----------------------'
    if opt == 'sgd':
        hype_print += '\n' + '--- SGD ---'
        hype_print += '\n' + 'Learning rate:' + str(lr)
        hype_print += '\n' + 'Decay: ' + str(decay)
        hype_print += '\n' + 'Momentum: ' + str(momentum)
        hype_print += '\n' + 'Nesterov: ' + str(nesterov)
        hype_print += '\n' + '-----------'
    elif opt == 'adam':
        hype_print += '\n' + '--- ADAM ---'
        hype_print += '\n' + 'lr: ' + str(a_lr)
        hype_print += '\n' + 'beta_1: ' + str(a_beta_1)
        hype_print += '\n' + 'beta_2: ' + str(a_beta_2)
        hype_print += '\n' + 'epsilon: ' + str(a_epsilon)
        hype_print += '\n' + 'decay: ' + str(a_decay)
        hype_print += '\n' + 'Amsgrad: ' + str(amsgrad)
        hype_print += '\n' + '------------'
    elif opt == 'rmsprop':
        hype_print += '\n' + '--- RMSprop ---'
        hype_print += '\n' + 'lr: ' + str(r_lr)
        hype_print += '\n' + 'rho: ' + str(r_rho)
        hype_print += '\n' + 'epsilon: ' + str(r_epsilon)
        hype_print += '\n' + 'decay: ' + str(r_decay)
        hype_print += '\n' + '------------'
    if lropf:
        hype_print += '\n' + '--- Reduce lr on plateau ---'
        hype_print += '\n' + 'lr decrease factor: ' + str(f_lrop)
        hype_print += '\n' + 'Patience: ' + str(p_lrop)
        hype_print += '\n' + 'Min delta: ' + str(md_lrop)
        hype_print += '\n' + 'Cool down:' + str(cd_lrop)
        hype_print += '\n' + 'Min lr: ' + str(mlr_lrop)
        hype_print += '\n' + '----------------------------'
    if sd:
        hype_print += '\n' + '--- Step decay ---'

    hype_print += '\n' + 'Workers: ' + str(workers)
    hype_print += '\n' + 'Shuffle: ' + str(shuffle)

    hype_print += '\n' + 'Number of training batches: ' + str(len(training_generator))

    if len(val_folders) > 0:
        hype_print += '\n' + 'Number of validation batches: ' + str(len(validation_generator))

    outcomes = 1
    # loss = 'mean_absolute_percentage_error'
    loss = 'mean_absolute_error'
    # loss = 'mean_squared_error'
    if feature == 'xy':
        outcomes = 2
        loss = 'mean_absolute_error'
        # loss = 'mean_squared_error'

    # keras.backend.set_image_data_format('channels_first')

    model, hype_print = regressor_selector(model_name, hype_print, channels, img_rows, img_cols, outcomes)

    hype_print += '\n' + '========================================================================================='

    # printing on screen hyperparameters
    print(hype_print)

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + feature + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)
    models_dir = join(root_dir, "models")
    mkdir(models_dir)
    # writing hyperparameters on file
    f = open(root_dir + '/hyperparameters.txt', 'w')
    f.write(hype_print)
    f.close()

    model.summary()

    callbacks = []

    if len(val_folders) > 0:
        checkpoint = ModelCheckpoint(
            filepath=models_dir + '/' + model_name + '_{epoch:02d}_{loss:.5f}_{val_loss:.5f}.h5', monitor='val_loss',
            save_best_only=True)
    else:
        checkpoint = ModelCheckpoint(
            filepath=models_dir + '/' + model_name + '_{epoch:02d}_{loss:.5f}.h5', monitor='loss',
            save_best_only=True)

    callbacks.append(checkpoint)

    # tensorboard = keras.callbacks.TensorBoard(log_dir=root_dir + "/logs",
    #                                          histogram_freq=5,
    #                                          batch_size=batch_size,
    #                                          write_images=True,
    #                                          update_freq=batch_size * 100)

    history = LossHistoryR()

    csv_callback = keras.callbacks.CSVLogger(root_dir + '/epochs_log.csv', separator=',', append=False)

    callbacks.append(history)
    callbacks.append(csv_callback)

    # callbacks.append(tensorboard)

    # sgd
    optimizer = None
    if opt == 'sgd':
        sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        optimizer = sgd
    elif opt == 'adam':
        adam = optimizers.Adam(lr=a_lr, beta_1=a_beta_1, beta_2=a_beta_2, epsilon=a_epsilon, decay=a_decay,
                               amsgrad=amsgrad)
        optimizer = adam
#    elif opt == 'adabound':
#        adabound = AdaBound(lr=ab_lr, final_lr=ab_final_lr, gamma=ab_gamma, weight_decay=ab_weight_decay,
#                            amsbound=False)
#        optimizer = adabound
    elif opt == 'rmsprop':
        rmsprop = optimizers.RMSprop(lr=r_lr, rho=r_rho, epsilon=r_epsilon, decay=r_decay)
        optimizer = rmsprop

    # reduce lr on plateau
    if lropf:
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=f_lrop, patience=p_lrop, verbose=1,
                                                 mode='auto',
                                                 min_delta=md_lrop, cooldown=cd_lrop, min_lr=mlr_lrop)
        callbacks.append(lrop)

    if sd:

        # learning rate schedule
        def step_decay(epoch):
            current = K.eval(model.optimizer.lr)
            lrate = current
            if epoch == 99:
                lrate = current / 10
                print('Reduced learning rate by a factor 10')
            return lrate

        stepd = LearningRateScheduler(step_decay)
        callbacks.append(stepd)

    if es:
        # early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=md_es, patience=p_es, verbose=1, mode='max')
        callbacks.append(early_stopping)

    if tb:
        tb_path = './tb/'
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        tb_path = os.path.join(tb_path, root_dir)
        os.mkdir(tb_path)
        tensorboard = TensorBoard(log_dir=tb_path)
        callbacks.append(tensorboard)

    model.compile(optimizer=optimizer, loss=loss)

    if len(val_folders) > 0:
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=len(training_generator),
                            validation_steps=len(validation_generator),
                            epochs=epochs,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)
    else:
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator),
                            epochs=epochs,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)

    # mp.set_start_method('fork')

    # save results
    train_history = root_dir + '/train-history'
    with open(train_history, 'wb') as file_pi:
        pickle.dump(history.dic, file_pi)

    # post training operations

    # training plots
    train_plots(train_history, False)

    if len(test_dirs) > 0:

        if len(val_folders) > 0:
            # get the best model on validation
            val_loss = history.dic['val_losses']
            m = val_loss.index(min(val_loss))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        else:
            # get the best model on validation
            acc = history.dic['losses']
            m = acc.index(min(acc))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        # test plots & results if test data is provided
        if len(test_dirs) > 0:
            pkl = tester(test_dirs, best, batch_size, time, feature, workers)
            test_plots(pkl, feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.', required=True)
    parser.add_argument(
        '-v', '--val_dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files valid data.', required=False)
    parser.add_argument(
        '--test_dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files test data.', required=False)
    parser.add_argument(
        '-m', '--model', type=str, default='', help='Model type.', required=True)
    parser.add_argument(
        '-e', '--epochs', type=int, default=10, help='Number of epochs.', required=True)
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=64, help='Batch size.', required=True)
    parser.add_argument(
        '-w', '--workers', type=int, default='', help='Specify number of workers.', required=True)
    parser.add_argument(
        '-opt', '--optimizer', type=str, default='adam', help='Specify the optimizer.', required=False)
    parser.add_argument(
        '-t', '--time', help='Feed the network with arrival time.', action="store_true")
    parser.add_argument(
        '-i', '--intensity_cut', type=float, default=50, help='Specify event intensity threshold (default 50 phe)', required=False)
    parser.add_argument(
        '--emin', type=float, default=-100, help='Specify min event MC energy in a log10 scale (default -100)',
        required=False)
    parser.add_argument(
        '--emax', type=float, default=100, help='Specify max event MC energy in a log10 scale (default +100)',
        required=False)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1e-04, help='Set Learning Rate (default 1e-04)', required=False)
    parser.add_argument(
        '--gpu_fraction', type=float, default=0.,
        help='Set limit to fraction of GPU memory usage. IMPORTANT: between 0 and 1.', required=False)
    parser.add_argument(
        '-f', '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)
    parser.add_argument(
        '--lrop', help='Reduce learning rate on plateau.', action="store_true")
    parser.add_argument(
        '--sd', help='Use step decay.', action="store_true")
    parser.add_argument(
        '--es', help='Use early stopping.', action="store_true")
    parser.add_argument(
        '--tb', help='Use TensorBoard.', action="store_true")

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    val_folders = FLAGS.val_dirs
    model_name = FLAGS.model
    time = FLAGS.time
    intens=FLAGS.intensity_cut
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    opt = FLAGS.optimizer
    lropf = FLAGS.lrop
    sd = FLAGS.sd
    es = FLAGS.es
    tb = FLAGS.tb
    gpufraction = FLAGS.gpu_fraction
    # we = FLAGS.iweights
    workers = FLAGS.workers
    test_dirs = FLAGS.test_dirs
    lr = FLAGS.learning_rate
    feature = FLAGS.feature
    emin = FLAGS.emin
    emax = FLAGS.emax

    regressor_training_main(folders, val_folders, model_name, time, epochs, batch_size, opt, lr, lropf, sd, es,
                            feature, workers, test_dirs, intens, tb, gpufraction, emin, emax)
