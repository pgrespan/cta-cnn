import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from ctapipe.instrument import CameraGeometry
# from ctapipe.visualization import CameraDisplay
from keras.models import load_model

from generators import DataGeneratorC
from utils import get_all_files


def tester(folders, mdl, batch_size, atime, workers):
    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorC(h5files, batch_size=batch_size, arrival_time=atime, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    pr_labels = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=workers,
                                        use_multiprocessing=True, verbose=1)

    test_idxs = test_generator.get_indexes()  # ground truth labels
    gt_labels = test_idxs[:, 2].reshape(test_idxs[:, 2].shape[0])
    pr_labels = pr_labels.reshape(pr_labels.shape[0])

    """ NON HA SENSO PERCHé SI STA FACENDO CONFRONTI TENENDO FISSATA LA SOGLIA 0.5...BISOGNA GUARDARE IN TUTTO IL RANGE
        DI SOGLIE, PER QUESTO SI USA LA ROC

    # get wrong predicted images
    diff = np.array(gt_labels) - np.around(pr_labels)
    wrong = np.nonzero(diff)

    test_idxs_wrong = test_idxs[wrong]

    sample_length = 1000

    if sample_length >= len(test_idxs_wrong):
        sample_length = len(test_idxs_wrong) - 1

    # choose randomly 1000 of them
    rnd_idxs = np.random.randint(len(test_idxs_wrong), size=sample_length)
    cento = test_idxs_wrong[rnd_idxs, :]
    # cento = np.random.choice(test_idxs_wrong, sample_length)

    
    # create pdf report
    nrow = sample_length
    ncol = 2
    geom = CameraGeometry.from_name("LSTCam")
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 1))

    for l, i in enumerate(cento):
        image, time, gt, mc_energy = test_generator.get_event(i)

        # image
        disp = CameraDisplay(geom, ax=axs[l, 0], title='GT: ' + str(gt))
        # disp.add_colorbar()
        disp.image = image

        # time
        disp = CameraDisplay(geom, ax=axs[l, 1], title='Energy: ' + str(mc_energy))
        # disp.add_colorbar()
        disp.image = time

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    fig.savefig(mdl + '_misc_report.pdf', format='pdf')
    
    

    # histogram based on failed predictions
    mis_en = np.array([])
    for l, i in enumerate(cento):
        image, time, gt, mc_energy = test_generator.get_event(i)
        mis_en = np.append(mis_en, [mc_energy])

    bins = 100

    plt.figure(0)
    plt.hist(mis_en, bins)
    plt.xlabel('Energy [TeV]')
    plt.ylabel('# of misclassified events [Log]')
    plt.yscale('log', nonposy='clip')
    plt.title('Misclassified events histogram - test set')

    plt.savefig(mdl + '_misc_hist.eps', format='eps')
    
    """

    # saving predictions
    df = pd.DataFrame()
    df['GroundTruth'] = gt_labels
    df['Predicted'] = pr_labels

    df.to_csv(mdl + '_test.csv', sep=',', index=False, encoding='utf-8')

    p_file = mdl + '_test.csv'

    print('Results saved in ' + p_file)

    return p_file


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

    FLAGS, unparsed = parser.parse_known_args()

    dirs = FLAGS.dirs
    m = FLAGS.model
    at = FLAGS.time
    bs = FLAGS.batch_size
    w = FLAGS.workers

    tester(dirs, m, bs, at, w)
