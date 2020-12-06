import argparse

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from generators import DataGeneratorChain
from utils import get_all_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model_sep', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_reg', type=str, default='', help='Path of the model to load.', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    test_dirs = FLAGS.test_dirs
    sep_path = FLAGS.model_sep
    reg_path = FLAGS.model_reg

    test_h5files = get_all_files(test_dirs)

    sep = load_model(sep_path)
    e_reg = load_model(reg_path)

    batch_size = 128
    print('Building test generator...')
    test_generator = DataGeneratorChain(test_h5files, batch_size=batch_size, arrival_time=True, shuffle=False)

    print('Inference on test data...')
    steps_done = 0
    steps = len(test_generator)

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 4)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, y, _, energy, _ = generator_output

        gammaness = sep.predict_on_batch(x)
        mc_energy_reco = e_reg.predict_on_batch(x)

        batch = np.concatenate((y, gammaness, energy, mc_energy_reco), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    cols = ['label', 'gammaness', 'mc_energy', 'mc_energy_reco']
    df_train = pd.DataFrame(table, columns=cols)

    df_train.to_pickle('gammaness_energy.pkl')

