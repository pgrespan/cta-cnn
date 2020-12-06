import argparse

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from tabulate import tabulate

from generators import DataGeneratorChain
from utils import get_all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=False)
    parser.add_argument(
        '--model_sep', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_energy', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_azalt', type=str, default='', help='Path of the model to load.', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    dirs = FLAGS.dirs
    time = FLAGS.time
    class_path = FLAGS.model_sep
    rege_path = FLAGS.model_energy
    altaz_path = FLAGS.model_azalt

    h5files = get_all_files(dirs)
    batch_size = 64

    classifier = load_model(class_path)
    reg_energy = load_model(rege_path)
    reg_direction = load_model(altaz_path)

    print('Building test generator...')
    test_generator = DataGeneratorChain(h5files, batch_size=batch_size, arrival_time=time, shuffle=True)

    # retrieve ground truth
    print('Inference on data...')
    steps_done = 0
    steps = len(test_generator)
    # steps = 2

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 9)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, y, intensity, energy, altaz = generator_output

        y_prd = classifier.predict_on_batch(x)
        e_reco = reg_energy.predict_on_batch(x)
        altaz_reco = reg_direction.predict_on_batch(x)

        # y_prd = y_prd.reshape(y_prd.shape[0])
        # e_reco = e_reco.reshape(e_reco.shape[0])

        # print('y: ', y.shape)
        # print('y_prd: ', y_prd.shape)
        # print('intensity: ', intensity.shape)
        # print('energy: ', energy.shape)
        # print('e_reco: ', e_reco.shape)
        # print('altaz: ', altaz.shape)
        # print('altaz_reco: ', altaz_reco.shape)

        alt = altaz[:, 0].reshape(altaz.shape[0], 1)
        az = altaz[:, 1].reshape(altaz.shape[0], 1)

        alt_reco = altaz_reco[:, 0].reshape(altaz_reco.shape[0], 1)
        az_reco = altaz_reco[:, 1].reshape(altaz_reco.shape[0], 1)

        batch = np.concatenate(
            (y, y_prd, intensity, energy, e_reco, alt, az, alt_reco, az_reco), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    cols = ['Label', 'gammanes', 'Intensity', 'mc_energy', 'mc_energy_reco', 'd_alt', 'd_az', 'd_alt_reco',
            'd_az_reco']
    df = pd.DataFrame(table, columns=cols)

    with open('lstch_analysis.txt', 'w') as f:
        print('CNN LST Chain - Full analysis\n', file=f)
        print('Separation Network:' + str(class_path) + '\n', file=f)
        print('Energy Network:' + str(rege_path) + '\n', file=f)
        print('Direction Network:' + str(altaz_path) + '\n', file=f)
        print('Number of files in the test set:' + str(len(h5files)) + '\n', file=f)
        print(tabulate(df, headers='keys', tablefmt='psql'), file=f)

    df.to_pickle('lstch_analysis.pkl')

    print(df)
