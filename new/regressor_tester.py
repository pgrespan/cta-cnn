import argparse

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from generators import DataGeneratorR
from utils import get_all_files
import tensorflow as tf
import keras.backend as K


def tester(folders, mdl, batch_size, time, feature, workers, intensity_cut, emin, emax):
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

    h5files = get_all_files(folders)
    # random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorR(h5files, feature=feature, batch_size=batch_size, arrival_time=time, shuffle=False, emin=emin, emax=emax)
    print('Number of test batches: ' + str(len(test_generator)))

    predict = model.predict_generator(generator=test_generator, max_queue_size=10, workers=workers,
                                      use_multiprocessing=True, verbose=1)

    # retrieve ground truth
    print('Retrieving ground truth...')
    gt_feature = []
    steps_done = 0
    steps = len(test_generator)

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=workers, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(output_generator)
        _, y = generator_output
        gt_feature.append(y)
        # print('steps_done', steps_done)
        # print(y)
        steps_done += 1
        progbar.update(steps_done)

    df = pd.DataFrame()

    if feature == 'energy':

        pr_feature = predict
        gt_feature = np.array(gt_feature).reshape(steps * batch_size)

        print('predict shape: ', predict.shape)
        print('gt_feature shape: ', gt_feature.shape)

        df['GroundTruth'] = gt_feature
        df['Predicted'] = pr_feature

    elif feature == 'xy':

        #REGRESSOR FOR ENERGY PREDITCION
        e_regr_path = '/home/nmarinel/ctasoft/cta-lstchain/cnn/ResNetFSE_44_0.12432_0.12943.h5'
        reg_energy = load_model(e_regr_path)

        pr_feature = predict
        gt_feature = np.array(gt_feature).reshape(steps * batch_size, 2)

        print('predict shape: ', predict.shape)
        print('gt_feature shape: ', gt_feature.shape)

        df['src_x'] = gt_feature[:, 0]
        df['src_y'] = gt_feature[:, 1]
        df['src_x_rec'] = pr_feature[:, 0]
        df['src_y_rec'] = pr_feature[:, 1]

        print('Building generator for energy...')
        energy_generator = DataGeneratorR(h5files, feature='energy', batch_size=batch_size, arrival_time=time,
                                          shuffle=False)

        # retrieve ground truth
        print('Retrieving ground truth...')
        gt_energy = []
        pr_energy = []
        steps_done = 0
        steps = len(energy_generator)

        enqueuer = OrderedEnqueuer(energy_generator, use_multiprocessing=True)
        enqueuer.start(workers=workers, max_queue_size=10)
        output_generator = enqueuer.get()

        progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            x, y = generator_output
            gt_energy.append(y)

            energy_reco = reg_energy.predict_on_batch(x)
            pr_energy.append(energy_reco)

            # print('steps_done', steps_done)
            # print(y)
            steps_done += 1
            progbar.update(steps_done)

        gt_energy = np.array(gt_energy).reshape(steps * batch_size)
        pr_energy = np.array(pr_energy).reshape(steps * batch_size)
        df['energy'] = gt_energy
        df['energy_reco'] = pr_energy

        print('gt_energy shape: ', gt_energy.shape)

    res_file = mdl + '_test_new.pkl'

    df.to_pickle(res_file)

    print('Results saved in ' + mdl + '_test_new.pkl')

    return res_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=False)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=True)
    parser.add_argument(
        '-i', '--intensity_cut', type=float, help='Specify event intensity threshold.', required=False)
    parser.add_argument(
        '--emin', type=float, default=-100, help='Specify min event MC energy in a log10 scale (default -100)',
        required=False)
    parser.add_argument(
        '--emax', type=float, default=100, help='Specify max event MC energy in a log10 scale (default +100)',
        required=False)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)

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

    tester(folders, model, batch_size, time, feature, workers, i, emin, emax)
