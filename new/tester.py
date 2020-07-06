import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
import keras
import utils
from lst_generator import LSTGenerator
import pandas as pd
import numpy as np
import argparse
from astropy import units as u
from ctapipe.coordinates import NominalFrame, AltAz

def tester(folders,
           model,
           batch_size,
           time,
           feature,
           workers,
           intensity_cut,
           emin,
           emax,
           class_model=''):

    ###################################
    # TensorFlow wizardry for GPU dynamic memory allocation
    #if gpu_fraction != 0 and gpu_fraction <= 1:
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a fraction of the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # Create a session with the above options specified.
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    all_files = utils.get_all_files(folders)

    print('Building test generator...')
    test_generator = LSTGenerator(
        h5files=all_files,
        feature=feature,
        batch_size=batch_size,
        arrival_time=time,
        shuffle=False,
        emin=emin,
        emax=emax,
        intensity=intensity_cut,
        class_model=class_model,
        gammaness=0.8
    )

    print('Number of test batches: ' + str(len(test_generator)))
    print('Loading model...')
    mdl = keras.models.load_model(model)
    print('Predicting {}...'.format(feature))
    predictions = mdl.predict_generator(generator=test_generator, max_queue_size=10, workers=workers,
                                      use_multiprocessing=True, verbose=1)

    # adjust length of indexes

    print('Retrieving true {}...'.format(feature))
    df = pd.DataFrame()
    df['energy_true'] = test_generator.get_energies(log=True)
    if feature =='gammaness':
        df['gammaness'] = predictions
    elif feature == 'energy':
        df['energy_reco'] = predictions
    elif feature == 'direction':
        df[['d_alt_true','d_az_true']] = test_generator.get_AltAz(nom_frame=True)
        df['d_alt_reco'] = predictions[:, 0]
        df['d_az_reco'] = predictions[:, 1]

    res_file = model + '_' + feature + '_test.pkl'
    df.to_pickle(res_file)
    print('Results saved in ' + res_file)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--time', help='Feed the network with arrival time.', action="store_true")
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=True)
    parser.add_argument(
        '-i', '--intensity_cut', type=float, help='Specify event intensity threshold.', required=False)
    parser.add_argument(
        '--emin', type=float, default=1e-1000, help='Specify min event MC energy.',
        required=False)
    parser.add_argument(
        '--emax', type=float, default=1e+1000, help='Specify max event MC energy.',
        required=False)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)
    parser.add_argument(
        '--class_model', type=str, default='', help='Classification model to evaluate gammaness.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    model = FLAGS.model
    time = FLAGS.time
    batch_size = FLAGS.batch_size
    workers = FLAGS.workers
    feature = FLAGS.feature
    i = FLAGS.intensity_cut
    emin = FLAGS.emin
    emax = FLAGS.emax
    #plot = FLAGS.save_plots
    class_model = FLAGS.class_model

    tester(
        folders=folders,
        model=model,
        batch_size=batch_size,
        time=time,
        feature=feature,
        workers=workers,
        intensity_cut=i,
        emin=emin,
        emax=emax,
        class_model=class_model
    )