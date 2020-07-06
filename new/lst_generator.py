import warnings
warnings.simplefilter('ignore')

import multiprocessing
import os

import h5py
import keras
import numpy as np
import pandas as pd

from astropy import units as u
from ctapipe.coordinates import NominalFrame, AltAz

class LSTGenerator(keras.utils.Sequence):
    'Generates data for Keras. NOTE: <shuffle=False> by default.'

    def __init__(
            self,
            h5files,
            feature,
            batch_size=32,
            arrival_time=False,
            shuffle=False,
            intensity=50,
            leakage2_intensity=0.2,
            emin=1e-1000,
            emax=1e+1000,
            class_model='',
            gammaness=0.7):

        self.batch_size = batch_size
        self.h5files = h5files
        self.feature = feature
        self.indexes = pd.DataFrame()
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.intensity = intensity
        self.leakage2_intensity = leakage2_intensity
        self.emin = emin
        self.emax = emax
        self.classify = False
        self.gammaness = gammaness
        self.apply_cuts()
        self.outcomes = 1
        if feature == 'direction':
            self.outcomes += 1
        if class_model != '':
            self.evaluate_gammaness(class_model) # also, sets self.classify to True
            self.apply_cuts()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch.'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data.'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch

        #indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size].reset_index(drop=True)
        indexes = self.indexes.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(indexes)
        # print("\n", index, "y shape: ", y.shape)
        # print("training idx: ", indexes)

        # if self.test_mode:
        #    self.feature_array = np.append(self.feature_array, y)

        return x, y

    def gamma_fraction(self):
        frac = np.mean(self.indexes['class'], axis = 0)
        return frac

    def set_feature(self, feature):
        '''Sets the feature (label y) that will be provided to the network by __getitem__().
        Choose between "gammaness", "energy" and "direction".'''
        self.feature = feature

    def evaluate_gammaness(self, model):
        'Gets predictions in order to perform a gammaness cut. ONLY for energy or direction reconstruction.'
        mdl = keras.models.load_model(model)
        feature_backup = self.feature

        # set generator to classification mode (maybe not needed?)
        self.set_feature('gammaness')

        # predict gammaness
        pr_labels = mdl.predict_generator(generator=self, steps=None, max_queue_size=10, workers=4,
                                            use_multiprocessing=True, verbose=1)
        # truncate indexes to the length of predictions. OBVIOUSLY TO BE FIXED
        self.indexes = self.indexes[:len(pr_labels)]
        self.indexes['gammaness'] = pr_labels
        # set back to regression mode, if needed (either energy reco or direction reco)
        self.set_feature(feature_backup)
        # set classify variable to true in order to apply cuts
        self.classify = True
        return pr_labels

    def get_all_info(
            self,
            e_min=None,
            e_max=None,
            alt_min=None,
            alt_max=None,
            az_min=None,
            az_max=None,
            int_min=None,
            int_max=None,
            leakage2_max=None
    ):
        'Returns a Pandas DataFrame containing all useful info about data (class, energy, intensity, direction...).'
        # why are they truncated? ---> 'cause # of events must be a multiple of batch_size (it seems so)!
        # num. of truncated << total num. btw...
        return self.indexes[0:self.__len__() * self.batch_size]

    def get_energies(self, log=True):
        'Set log to True (default) to get the log10 of energies (TeV)'
        if log:
            return np.log10(self.get_all_info()['energy_true'])
        else:
            return self.get_all_info()['energy_true']

    def get_AltAz(self, nom_frame=True):
        'Set nom_frame to True (default) to get source direction in Nominal Frame.'
        if nom_frame:
            return self.get_all_info()[['d_alt_true','d_az_true']]
        else:
            return self.get_all_info()[['alt','az']]

    def get_class(self):
        'Gamma or proton?'
        return self.get_all_info()['class']

    def get_charges(self, info, original=True):
        'Returns charge images of selected data. DO NOT USE FOR MANY EVENTS: will overload memory (to be fixed)'
        # down = idx*self.batch_size
        # up = (idx+1)*self.batch_size
        # if up >= len(info):
        #     up = len(info) - 1
        # indexes = info[idx*self.batch_size:(idx+1)*self.batch_size]
        x, y = self.__data_generation(indexes = info, original=original, unset_time=True)
        return x, y

    def apply_cuts(self):
        self.indexes = self.indexes[
            (self.indexes['intensity'] >= self.intensity) &
            (self.indexes['leakage2'] <= self.leakage2_intensity) &
            (self.indexes['energy_true'] <= self.emax) &
            (self.indexes['energy_true'] >= self.emin)
        ]

        if self.classify:
            self.indexes = self.indexes[self.indexes['gammaness'] >= self.gammaness]

        self.indexes.index = pd.RangeIndex(len(self.indexes)) # same as self.indexes = self.indexes.reset_index(drop=True), but faster

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        #idx = np.array([], dtype=np.int64).reshape(0, 8)
        df = pd.DataFrame()
        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            idx = pd.DataFrame()
            idx['event_index'] = h5f['LST/LST_event_index'][:]
            idx['file_index'] = [positions[l]]*len(idx['event_index'])
            idx['image_index'] = np.arange(len(idx['event_index']))
            energy_MC = []
            az = []
            alt = []
            for id in idx['event_index']:
                energy_MC.append(h5f['Event_Info/ei_mc_energy'][:][int(id)])
                az.append(h5f['Event_Info/ei_az'][:][int(id)])
                alt.append(h5f['Event_Info/ei_alt'][:][int(id)])
            idx['energy_true'] = np.array(energy_MC) # AAA: the network will be fed with LOG10(energy_true)
            idx['alt'] = np.degrees(np.array(alt))
            idx['az'] = np.degrees(np.array(az))
            idx['d_alt_true'], idx['d_az_true'] = self.to_nominal_frame(idx)
            idx['intensity'] = h5f['LST/intensities'][:]
            idx['leakage2'] = h5f['LST/intensities_width_2'][:]
            fn_basename = os.path.basename(os.path.normpath(f))
            idx['class'] = np.zeros(len(idx['image_index']))  # class: proton by default
            if fn_basename.startswith('g'):
                idx['class'] = np.ones(len(idx['image_index'])) # if filename begins with 'g' ('gamma...'), switch to class gamma [1]
            h5f.close()
            #idx['index'] = np.arange(len(idx['event_index']))
            #cp = np.dstack(([positions[l]] * len(r), r, event_idx, intensities, intensities_width_2, energy_MC, alt, az)).reshape(-1, 8)

            df = df.append(idx)
        return_dict[i] = df

    def generate_indexes(self):

        cpu_n = multiprocessing.cpu_count()
        pos = self.chunkit(np.arange(len(self.h5files)), cpu_n)
        h5f = self.chunkit(self.h5files, cpu_n)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        processes = []

        if cpu_n >= len(self.h5files):
            # print('ncpus >= num_files')
            for i, f in enumerate(self.h5files):
                p = multiprocessing.Process(target=self.worker, args=([f], [i], i, return_dict))
                p.start()
                processes.append(p)
        else:
            # print('ncpus < num_files')
            for i in range(cpu_n):
                p = multiprocessing.Process(target=self.worker, args=(h5f[i], pos[i], i, return_dict))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        for key, value in return_dict.items():
            self.indexes = self.indexes.append(value, ignore_index=True)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.indexes = self.indexes.sample(frac=1).reset_index(drop=True)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def to_nominal_frame(self, indexes):

        alt_LST = 70  # deg
        az_LST = 180  # deg


        point = AltAz(
            alt=alt_LST * u.deg,
            az=az_LST * u.deg
        )
        alt = np.array(indexes['alt']) # altitude
        az = np.array(indexes['az']) # azimuth
        # print("\nalt shape: {}, az shape: {}".format(indexes['alt'].shape, indexes['az'].shape))
        src = AltAz(
            alt=alt * u.deg,
            az=az * u.deg
        )
        source_direction = src.transform_to(NominalFrame(origin=point))
        delta_alt = source_direction.delta_alt.deg
        delta_az = source_direction.delta_az.deg

        return delta_alt, delta_az

    def __data_generation(self, indexes, original=False, unset_time=False):
        'Generates data containing batch_size samples'
        # Initialization
        IMG_ROWS = 100 #96
        IMG_COLS = 100 #88
        length = len(indexes)
        if unset_time:
            arrival_time = False
        else:
            arrival_time = self.arrival_time
        x = np.empty([length, IMG_ROWS, IMG_COLS, arrival_time + 1])
        y = np.empty([length, self.outcomes], dtype=float)

        if original:
            CRG_LEN = 1855
            x = np.empty([length, CRG_LEN])

        if self.feature == 'gammaness':
            # Store class
            y = np.array(indexes['class']) # MAYBE it needs a rehsape
        if self.feature == 'energy':
            # Store energy
            y = np.log10(np.array( indexes['energy_true'] )) # MC energy in LOG10 / MAYBE needs a reshape
        elif self.feature == 'direction':
            # Store direction
            y[:, 0] = np.array( indexes['d_alt_true'] )
            y[:, 1] = np.array( indexes['d_az_true'] )

        # Generate data
        for i, (_, row) in enumerate(indexes.iterrows()):

            filename = self.h5files[int(row['file_index'])]

            h5f = h5py.File(filename, 'r')

            # Store images
            if original:
                x[i,:] = h5f['LST/LST_image_charge'][int(row['image_index'])]
            else:
                x[i, :, :, 0] = h5f['LST/LST_image_charge_interp'][int(row['image_index'])]
                if arrival_time:
                    x[i, :, :, 1] = h5f['LST/LST_image_peak_times_interp'][int(row['image_index'])]

            h5f.close()

        # x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y