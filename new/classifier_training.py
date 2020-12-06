import warnings
warnings.simplefilter('ignore')

import argparse
import datetime
import multiprocessing as mp
import os
import pickle
from os import listdir
from os import mkdir
from os.path import isfile, isdir, join
from pathlib import Path
import keras
import keras.backend as K
import numpy as np
from lst_generator import LSTGenerator
from classifier_selector import select_classifier
from classifier_test_plots import test_plots
from classifier_tester import tester
from classifier_training_plots import train_plots
from generators import DataGeneratorC
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
from losseshistory import LossHistoryC
from clr import LRFinder
from utils import get_all_files
from keras_contrib.callbacks import CyclicLR

import tensorflow as tf


def classifier_training_main(
        folders,
        val_folders,
        model_name,
        time,
        epochs,
        batch_size,
        opt,
        learn_rate,
        lropf=False,
        sd=False,
        es=False,
        clr=False,
        workers=1,
        test_dirs='',
        load_model=False,
        tb=False,
        intensity_cut=None,
        leakage=0.2,
        gpu_fraction=1,
        train_indexes=None,
        valid_indexes=None,
        clr_values=[5e-5, 5e-3, 4]):
    ###################################
    # TensorFlow wizardry for GPU dynamic memory allocation
    if 0 < gpu_fraction < 1:
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Only allow a fraction of the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    # remove semaphore warnings
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    # avoid validation deadlock problem
    mp.set_start_method('spawn', force=True)

    # hard coded parameters
    shuffle = True
    channels = 1
    if time:
        channels = 2

    # early stopping
    md_es = 0.01  # min delta
    p_es = 25  # patience

    # cycle learning rate CLR
    base_lr = clr_values[0]
    max_lr = clr_values[1]
    step_size = clr_values[2]

    # sgd
    lr = 0.01  # lr
    decay = 1e-4  # decay
    momentum = 0.9  # momentum
    nesterov = True

    # adam
    # default a_lr should be 0.001
    a_lr = learn_rate
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
    amsbound = True

    # reduce lr on plateau
    f_lrop = 0.1  # factor
    p_lrop = 15  # patience
    md_lrop = 0.005  # min delta
    cd_lrop = 5  # cool down
    mlr_lrop = a_lr / 100  # min lr

    # cuts
    #intensity_cut = 500
    #leakage2_intensity_cut = 0.2

    training_files = get_all_files(folders)
    validation_files = get_all_files(val_folders)

    # generators
    print('Building training generator...')
    feature = 'gammaness'  # hardcoded by now
    '''
    training_generator = DataGeneratorC(training_files,
                                        batch_size=batch_size,
                                        arrival_time=time,
                                        shuffle=shuffle,
                                        intensity=intensity_cut)
    train_idxs = training_generator.get_indexes()
    train_gammas = np.unique(train_idxs[:, 2], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs[:, 2], return_counts=True)[1][0]
    '''
    training_generator = LSTGenerator(training_files,
                                      batch_size=batch_size,
                                      arrival_time=time,
                                      feature=feature,
                                      shuffle=shuffle,
                                      intensity=intensity_cut,
                                      leakage2_intensity=leakage,
                                      load_indexes=train_indexes)
    # get image size (rows and columns)
    img_rows = training_generator.img_rows
    img_cols = training_generator.img_cols
    print("IMG rows: {}, cols: {}".format(img_rows, img_cols))
    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)
    models_dir = join(root_dir, "models")
    mkdir(models_dir)

    # save data info
    train_idxs = training_generator.get_all_info()
    train_gammas = np.unique(train_idxs['class'], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs['class'], return_counts=True)[1][0]
    train_gamma_frac = training_generator.gamma_fraction()
    if train_indexes is None:
        train_idxs.to_pickle(join(root_dir, "train_indexes.pkl"))

    if len(val_folders) > 0:
        print('Building validation generator...')
        validation_generator = LSTGenerator(validation_files,
                                            batch_size=batch_size,
                                            arrival_time=time,
                                            feature=feature,
                                            shuffle=False,
                                            intensity=intensity_cut,
                                            leakage2_intensity=leakage,
                                            load_indexes=valid_indexes
                                            )
        valid_idxs = validation_generator.get_all_info()
        valid_gammas = np.unique(valid_idxs['class'], return_counts=True)[1][1]
        valid_protons = np.unique(valid_idxs['class'], return_counts=True)[1][0]
        valid_gamma_frac = validation_generator.gamma_fraction()
        if valid_indexes is None:
            valid_idxs.to_pickle(join(root_dir, "valid_indexes.pkl"))

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
    hype_print += '\n' + 'Validation: ' + str(val_folders)
    hype_print += '\n' + 'Test dirs: ' + str(test_dirs)

    hype_print += '\n' + 'intensity_cut: ' + str(intensity_cut)
    hype_print += '\n' + 'leakage2_intensity_cut: ' + str(leakage)

    if clr:
        hype_print += '\n' + '--- Cycle Learning Rate ---'
        hype_print += '\n' + 'Base LR: ' + str(base_lr)
        hype_print += '\n' + 'Max LR: ' + str(max_lr)
        hype_print += '\n' + 'Step size: ' + str(step_size) + ' (' + str(step_size*len(training_generator)) + ')'
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
    hype_print += '\n' + 'Number of training gammas: ' + str(train_gammas)
    hype_print += '\n' + 'Number of training protons: ' + str(train_protons)
    hype_print += '\n' + 'Fraction of gamma in training set: ' + str(train_gamma_frac)
    if len(val_folders) > 0:
        hype_print += '\n' + 'Number of validation batches: ' + str(len(validation_generator))
        hype_print += '\n' + 'Number of validation gammas: ' + str(valid_gammas)
        hype_print += '\n' + 'Number of validation protons: ' + str(valid_protons)
        hype_print += '\n' + 'Fraction of gamma in validation set: ' + str(valid_gamma_frac)

    # keras.backend.set_image_data_format('channels_first')
    if load_model:
        model = keras.models.load_model(model_name)
        model_name = Path(model_name).name
    else:
        model, hype_print = select_classifier(model_name, hype_print, channels, img_rows, img_cols)
    #model = load_model('/home/pgrespan/nick_models/ResNetFSE_49_0.82375_0.80292.h5')

    hype_print += '\n' + '========================================================================================='

    # printing on screen hyperparameters
    print(hype_print)

    # writing hyperparameters on file
    f = open(root_dir + '/hyperparameters.txt', 'w')
    f.write(hype_print)
    f.close()

    model.summary()

    callbacks = []

    if len(val_folders) > 0:
        checkpoint = ModelCheckpoint(
            filepath=models_dir + '/' + model_name + '_{epoch:02d}_{acc:.5f}_{val_acc:.5f}.h5', monitor='val_acc',
            save_best_only=False)
    else:
        checkpoint = ModelCheckpoint(
            filepath=models_dir + '/' + model_name + '_{epoch:02d}_{acc:.5f}.h5', monitor='acc',
            save_best_only=True)

    callbacks.append(checkpoint)

    # tensorboard = keras.callbacks.TensorBoard(log_dir=root_dir + "/logs",
    #                                          histogram_freq=5,
    #                                          batch_size=batch_size,
    #                                          write_images=True,
    #                                          update_freq=batch_size * 100)

    history = LossHistoryC()

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
    '''
    elif opt == 'adabound':
        adabound = AdaBound(lr=ab_lr, final_lr=ab_final_lr, gamma=ab_gamma, weight_decay=ab_weight_decay,
                            amsbound=False)
        optimizer = adabound
    '''
    # reduce lr on plateau
    if lropf:
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=f_lrop, patience=p_lrop, verbose=1,
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
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=md_es, patience=p_es, verbose=1, mode='max')
        callbacks.append(early_stopping)

    if tb:
        tb_path = os.path.join(root_dir, 'tb')
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        #tb_path = os.path.join(tb_path, root_dir)
        # os.mkdir(tb_path)
        tensorboard = TensorBoard(log_dir=tb_path)
        callbacks.append(tensorboard)

    if clr:
        cyclelr = CyclicLR(
            base_lr=base_lr,
            max_lr=max_lr,
            step_size=step_size*len(training_generator)
        )
        callbacks.append(cyclelr)

    lrfinder=False
    if lrfinder:

        lr_callback = LRFinder(len(training_generator)*batch_size, batch_size,
                               1e-05, 1e01,
                               # validation_data=(X_val, Y_val),
                               lr_scale='exp', save_dir=join(root_dir,'clr'))
        callbacks.append(lr_callback)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    if len(val_folders) > 0:
        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            steps_per_epoch=len(training_generator),
            validation_steps=len(validation_generator),
            epochs=epochs,
            verbose=1,
            max_queue_size=10,
            use_multiprocessing=True,
            workers=workers,
            shuffle=False,
            callbacks=callbacks
        )
    else:
        model.fit(
            x=training_generator,
            steps_per_epoch=len(training_generator),
            epochs=epochs,
            verbose=1,
            max_queue_size=10,
            use_multiprocessing=True,
            workers=workers,
            shuffle=False,
            callbacks=callbacks
        )

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
            val_acc = history.dic['val_accuracy']
            m = val_acc.index(max(val_acc))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        else:
            # get the best model
            acc = history.dic['accuracy']
            m = acc.index(max(acc))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        # test plots & results if test data is provided
        if len(test_dirs) > 0:
            csv = tester(test_dirs, best, batch_size, time, workers)
            test_plots(csv)


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
        '-i', '--intensity_cut', type=float, default=None, help='Specify event intensity threshold (default 50 phe)', required=False)
    parser.add_argument(
        '-lkg', '--leakage2', type=float, default=None, help='Specify event leakage2 max value (default 0.2)',
        required=False)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1e-04, help='Set Learning Rate (default 1e-04)', required=False)
    parser.add_argument(
        '--lrop', help='Reduce learning rate on plateau.', action="store_true")
    parser.add_argument(
        '--sd', help='Use step decay.', action="store_true")
    parser.add_argument(
        '--es', help='Use early stopping.', action="store_true")
    parser.add_argument(
        '--clr', help='Use Cycle Learning Rate policy.', action="store_true")
    parser.add_argument(
        '--clr_values', help='CLR values.', type=float, nargs=3, default=[1e-4, 1e-3, 3], required=False)
    parser.add_argument(
        '--tb', help='Use TensorBoard.', action="store_true")
    parser.add_argument(
        '--gpu_fraction', type=float, default=0., help='Set limit to fraction of GPU memory usage. IMPORTANT: between 0 and 1.', required=False)
    parser.add_argument(
        '--load_model', help='Select option in order to load a previously trained model.', action='store_true')
    parser.add_argument(
        '--train_indexes', type=str, default=None, help='Load training indexes.', required=False)
    parser.add_argument(
        '--valid_indexes', type=str, default=None, help='Load validation indexes.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    val_folders = FLAGS.val_dirs
    model_name = FLAGS.model
    time = FLAGS.time
    intens=FLAGS.intensity_cut
    lkg = FLAGS.leakage2
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    opt = FLAGS.optimizer
    lropf = FLAGS.lrop
    sd = FLAGS.sd
    es = FLAGS.es
    tb = FLAGS.tb
    gpu_fraction = FLAGS.gpu_fraction
    # we = FLAGS.iweights
    workers = FLAGS.workers
    test_dirs = FLAGS.test_dirs
    lr = FLAGS.learning_rate
    load_model=FLAGS.load_model
    clr=FLAGS.clr
    clr_values = FLAGS.clr_values
    train_indexes=FLAGS.train_indexes
    valid_indexes = FLAGS.valid_indexes

    classifier_training_main(
        folders=folders,
        val_folders=val_folders,
        load_model=load_model,
        model_name=model_name,
        time=time,
        epochs=epochs,
        batch_size=batch_size,
        opt=opt,
        learn_rate=lr,
        lropf=lropf,
        sd=sd,
        es=es,
        clr=clr,
        workers=workers,
        test_dirs=test_dirs,
        tb=tb,
        intensity_cut=intens,
        leakage=lkg,
        gpu_fraction=gpu_fraction,
        train_indexes=train_indexes,
        valid_indexes=valid_indexes,
        clr_values=clr_values
    )
