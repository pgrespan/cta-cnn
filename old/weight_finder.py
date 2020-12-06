import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize

from generators import DataGeneratorR
from utils import get_all_files


def retrieve_energy(generator):
    # retrieve energy
    print('Retrieving ground truth...')
    gt_feature = []
    steps_done = 0
    steps = len(generator)

    enqueuer = OrderedEnqueuer(generator, use_multiprocessing=True)
    enqueuer.start(workers=24, max_queue_size=10)
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

    return np.array(gt_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dir', type=str, default='', nargs='+', help='Folder that contain .h5 files.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    folder = FLAGS.dir

    f_basename = os.path.basename(os.path.normpath(folder[0]))

    h5files = get_all_files(folder)

    generator = DataGeneratorR(h5files, feature='energy', batch_size=64, arrival_time=False, shuffle=False)

    ei_mc_energy = retrieve_energy(generator)

    ei_mc_energy = np.array(ei_mc_energy).reshape(64 * len(generator))

    hist = np.histogram(ei_mc_energy, bins=50)
    edges = hist[1]

    inds = np.digitize(ei_mc_energy, edges)

    class_weights = compute_class_weight('balanced', np.unique(inds), inds)
    class_weights = class_weights / np.linalg.norm(class_weights)

    np.savez(f_basename + '_intra-class_weights.npz', edges=np.array(edges),
             class_weights=np.array(class_weights))

    plt.plot(class_weights)
    plt.yscale('log', nonposy='clip')
    plt.title(f_basename + '\nintra-class weights')
    plt.xlabel('Digitized energy')
    plt.ylabel('Weight')
    plt.savefig(f_basename + '_intra-class_weights.png', format='png')

