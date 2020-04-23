import argparse
import random
import keras.backend as K
import pandas as pd
from keras.models import load_model

from generators import DataGeneratorC
from utils import get_all_files
import tensorflow as tf

gpu_fraction = 0.5
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a fraction of the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def tester(folders, mdl, batch_size, atime, workers, intensity):
    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorC(h5files, batch_size=batch_size, arrival_time=atime, shuffle=False, intensity=intensity)
    print('Number of test batches: ' + str(len(test_generator)))

    pr_labels = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=workers,
                                        use_multiprocessing=True, verbose=1)

    test_idxs = test_generator.get_indexes()  # ground truth labels
    gt_labels = test_idxs[:, 2].reshape(test_idxs[:, 2].shape[0])
    pr_labels = pr_labels.reshape(pr_labels.shape[0])

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
    parser.add_argument(
        '-i', '--intensity_cut', type=float, help='Specify event intensity threshold.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    dirs = FLAGS.dirs
    m = FLAGS.model
    at = FLAGS.time
    bs = FLAGS.batch_size
    w = FLAGS.workers
    i = FLAGS.intensity_cut

    tester(dirs, m, bs, at, w, i)
