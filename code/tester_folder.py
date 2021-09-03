import warnings
warnings.simplefilter('ignore')

from tester import tester
import argparse
from utils import get_all_networks
def tester_folder(
           cnn_dirs,
           folders,
           batch_size,
           time,
           feature,
           workers,
           intensity_cut,
           leakage,
           gpu_fraction,
           test_indexes=None):

    cnns = get_all_networks(cnn_dirs)

    for model in cnns:
        tester(
            folders=folders,
            model=model,
            batch_size=batch_size,
            time=time,
            feature=feature,
            workers=workers,
            intensity_cut=intensity_cut,
            leakage=leakage,
            gpu_fraction=gpu_fraction,
            test_indexes=test_indexes
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cnn_folder', type=str, default='', nargs='+', help='Folder that contain .h5 networks.', required=True)
    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--time', help='Feed the network with arrival time.', action="store_true")
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=False)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=False)
    parser.add_argument(
        '-i', '--intensity_cut', type=float, default=None, help='Specify event intensity threshold.', required=False)
    parser.add_argument(
        '-lkg', '--leakage2', type=float, default=None, help='Specify event max leakage', required=False)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)
    parser.add_argument(
        '--gpu_fraction', type=float, default=1,
        help='Set limit to fraction of GPU memory usage. IMPORTANT: between 0 and 1.', required=False)
    parser.add_argument(
        '--test_indexes', type=str, default=None, help='Load test indexes.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    cnn_folder = FLAGS.cnn_folder
    folders = FLAGS.dirs
    time = FLAGS.time
    batch_size = FLAGS.batch_size
    workers = FLAGS.workers
    feature = FLAGS.feature
    i = FLAGS.intensity_cut
    lkg = FLAGS.leakage2
    gpufraction = FLAGS.gpu_fraction
    test_indexes = FLAGS.test_indexes


    tester_folder(
        cnn_dirs=cnn_folder,
        folders=folders,
        batch_size=batch_size,
        time=time,
        feature=feature,
        workers=workers,
        intensity_cut=i,
        leakage=lkg,
        gpu_fraction=gpufraction,
        test_indexes=test_indexes
    )

