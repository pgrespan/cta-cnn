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

class FullEventReconstructor:

    def __init__(
            self,
            folders,
            time,
            batch_size=128,
            workers=8,
            intensity_cut=None,
            leakage=None,
            output_path=None,
            gpu_fraction=1,
            test_indexes=None,
            plike = False):

        self.folders=folders
        self.batch_size = batch_size
        self.workers = workers
        self.intensity_cut = intensity_cut
        self.leakage = leakage
        self.gpu_fraction = gpu_fraction
        self.time = time
        print("Time: {}".format(time))
        self.test_indexes = test_indexes
        self.setGpuFraction()
        self.plike = plike

    def setGpuFraction(self):
        if 0 >= self.gpu_fraction or self.gpu_fraction > 1:
            pass
        ###################################
        #TensorFlow wizardry for GPU dynamic memory allocation
        else:
            config = tf.ConfigProto()
            # Don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True
            # Only allow a fraction of the GPU memory to be allocated
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            # Create a session with the above options specified.
            keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            ###################################

    def predictor(self, model, generator, feature):
        print('Loading model for {} reconstruction...'.format(feature))
        mdl = keras.models.load_model(model)
        print('Predicting {}...'.format(feature))
        predictions = mdl.predict_generator(
            generator=generator,
            max_queue_size=10,
            workers=self.workers,
            use_multiprocessing=True,
            verbose=1
        )
        return predictions

    def reconstructFullEvent(self, sep_model, ene_model, dir_model, output_path=None):
        all_files = utils.get_all_files(folders)

        print('Building test generator...')

        test_generator = LSTGenerator(
            h5files=all_files,
            feature="energy", # could be any of the 3
            batch_size=self.batch_size,
            arrival_time=self.time,
            shuffle=False,
            intensity=self.intensity_cut,
            leakage2_intensity=self.leakage,
            plike=self.plike,
            load_indexes=self.test_indexes
        )

        if test_indexes is None:
            test_idxs = test_generator.get_all_info()
            test_idxs.to_pickle("./test_indexes.pkl")
            pd.DataFrame(all_files).to_csv("./test_files.csv")

        print('Number of test batches: ' + str(len(test_generator)))

        # predict and save
        df = pd.DataFrame()

        print('Retrieving true gammaness...')
        df['class'] = test_generator.get_class()
        df['gammaness'] = self.predictor(sep_model, test_generator, "gammaness")

        print('Retrieving true energy...')
        df['energy_true'] = test_generator.get_energies(log=True)
        df['energy_reco'] = self.predictor(ene_model, test_generator, "energy")

        print('Retrieving true direction...')
        df['d_alt_true'] = test_generator.get_AltAz(nom_frame=True)[:, 0]
        df['d_az_true'] = test_generator.get_AltAz(nom_frame=True)[:, 1]
        reco_dir = self.predictor(dir_model, test_generator, "direction")
        df['d_alt_reco'] = reco_dir[:, 0]
        df['d_az_reco'] = reco_dir[:, 1]

        if (output_path == None):
            output_path = './i{}_l{}_fullEventReco.pkl'.format(int(self.intensity_cut), self.leakage)
        df.to_pickle(output_path)
        print('Results saved in ' + output_path)
        #print('Summary: \nTest gammas: {} \nTest protons: {}, \nGamma fraction: {}'.format(test_gammas, test_protons, test_gamma_frac))
        return df

    def predictFolders(self, sep_path, ene_path, dir_path, output_path):
        s_models = utils.get_all_networks([sep_path])
        e_models = utils.get_all_networks([ene_path])
        d_models = utils.get_all_networks([dir_path])

        length = min(len(s_models), len(e_models), len(d_models))

        for i in range(length):
            output = output_path + "_{}.h5".format(i)
            self.reconstructFullEvent(s_models[i], e_models[i], d_models[i], output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '-s', '--sep_path', type=str, default='', help='Folder of the g/h separators.', required=True)
    parser.add_argument(
        '-e', '--ene_path', type=str, default='', help='Folder of the energy regressors.', required=True)
    parser.add_argument(
        '-d', '--dir_path', type=str, default='', help='Folder of the direction regressors.', required=True)
    parser.add_argument(
        '-o', '--output', type=str, default=None, help='Output file (full path)', required=False)
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
        '--gpu_fraction', type=float, default=0.25,
        help='Set limit to fraction of GPU memory usage. IMPORTANT: between 0 and 1.', required=False)
    parser.add_argument(
        '--test_indexes', type=str, default=None, help='Load test indexes.', required=False)
    parser.add_argument(
        '--pointlike', help='Feed the network with arrival time.', action="store_true")

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    s_path = FLAGS.sep_path
    e_path = FLAGS.ene_path
    d_path = FLAGS.dir_path
    output = FLAGS.output
    time = FLAGS.time
    batch_size = FLAGS.batch_size
    workers = FLAGS.workers
    i = FLAGS.intensity_cut
    lkg = FLAGS.leakage2
    gpufraction = FLAGS.gpu_fraction
    test_indexes = FLAGS.test_indexes
    plike = FLAGS.pointlike
    fer = FullEventReconstructor(
        folders=folders,
        batch_size=batch_size,
        time=time,
        workers=workers,
        intensity_cut=i,
        leakage=lkg,
        gpu_fraction=gpufraction,
        test_indexes=test_indexes,
        plike=plike
    )

    fer.predictFolders(s_path, e_path, d_path, output_path=output)