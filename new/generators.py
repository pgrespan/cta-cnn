import warnings
warnings.simplefilter('ignore')

import multiprocessing
import os

import h5py
import keras
import numpy as np
import pandas as pd

from ctapipe.image import hillas_parameters, tailcuts_clean, leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.cleaning import number_of_islands
from astropy.coordinates import AltAz
from lstchain.reco.utils import sky_to_camera #, disp_parameters
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from ctapipe.coordinates import NominalFrame, AltAz


class DataGeneratorC(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, batch_size=32, arrival_time=False, shuffle=True, intensity=50, leakage2_intensity=0.2, emin=-100, emax=100):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 7)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.intensity = intensity
        self.leakage2_intensity = leakage2_intensity
        self.emin = emin
        self.emax = emax
        self.apply_cuts()
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch,
        # i.e. total number of images in the dataset / batch size
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):

        # print("training idx: ", index, '/', self.__len__())

        # with self.lock:
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(indexes)

        # print("training idx: ", indexes)
        return x, y

    def gamma_fraction(self):
        frac = np.mean(self.indexes, axis = 0, dtype=np.float64)[2]
        return frac

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

    def apply_cuts(self):
        self.indexes = self.indexes[
            (self.indexes[:, 4] >= self.intensity) &
            (self.indexes[:, 5] <= self.leakage2_intensity) &
            (self.indexes[:, 6] <= self.emax) &
            (self.indexes[:, 6] >= self.emin)
            ]

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 7)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            intensities = h5f['LST/intensities'][:]
            intensities_width_2 = h5f['LST/intensities_width_2'][:]
            lst_idx = h5f['LST/LST_event_index'][:]
            energy_MC = []
            for id in lst_idx:
                energy_MC.append(np.log10(h5f['Event_Info/ei_mc_energy'][:][int(id)]))
            energy_MC = np.array(energy_MC)
            h5f.close()
            r = np.arange(len(lst_idx))

            fn_basename = os.path.basename(os.path.normpath(f))

            clas = np.zeros(len(r))  # class: proton by default

            if fn_basename.startswith('g'):
                clas = np.ones(len(r))

            # cp = np.dstack(np.meshgrid([positions[l]], r, clas, lst_idx)).reshape(-1, 4)  # cartesian product
            cp = np.dstack(([positions[l]] * len(r), r, clas, lst_idx, intensities, intensities_width_2, energy_MC)).reshape(-1, 7)

            idx = np.append(idx, cp, axis=0)
        return_dict[i] = idx

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
            self.indexes = np.append(self.indexes, value, axis=0)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, 100, 100, self.arrival_time + 1])
        y = np.empty([self.batch_size], dtype=int)

        # print('__data_generation', indexes)

        # Generate data

        for i, row in enumerate(indexes):

            # print(row[0])

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')
            # Store image
            x[i, :, :, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, :, :, 1] = h5f['LST/LST_image_peak_times_interp'][int(row[1])]
            # Store class
            y[i] = int(row[2])

            h5f.close()

        return x, y


class DataGeneratorR(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, feature, batch_size=32, arrival_time=False, shuffle=True, intensity=50, leakage2_intensity=0.2, emin=-1000, emax=1000):
        self.batch_size = batch_size
        self.h5files = h5files
        self.feature = feature
        self.indexes = pd.DataFrame()
        #self.indexes = np.array([], dtype=np.int64).reshape(0, 8)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.intensity = intensity
        self.leakage2_intensity = leakage2_intensity
        self.emin = emin
        self.emax = emax
        self.apply_cuts()
        self.outcomes = 1
        if feature == 'direction': self.outcomes += 1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(indexes)

        # print("training idx: ", indexes)

        # if self.test_mode:
        #    self.feature_array = np.append(self.feature_array, y)

        return x, y

    def get_ordered_info(
            self,
            emin=None,
            emax=None,
            altmin=None,
            altmax=None,
            azmin=None,
            azmax=None,
            imin=None,
            imax=None,
            lkg=None
    ):
        # perché esplicitare? Forse perché alcuni elementi vengono troncati dall'approssimazione?
        # return self.indexes[0:self.__len__() * self.batch_size]

        # Così lo faccio io:
        return self.indexes

    def apply_cuts(self):
        self.indexes = self.indexes[
            (self.indexes['intensity'] >= self.intensity) &
            (self.indexes['leakage2'] <= self.leakage2_intensity) &
            (self.indexes['energy_true'] <= self.emax) &
            (self.indexes['energy_true'] >= self.emin) #&
            #(self.indexes['alt'] >= self.altmin) &
            #(self.indexes['alt'] <= self.altmax) &
            #(self.indexes['az'] >= self.azmin) &
            #(self.indexes['az'] <= self.azmax)
        ]

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
                energy_MC.append(np.log10(h5f['Event_Info/ei_mc_energy'][:][int(id)]))
                az.append(h5f['Event_Info/ei_az'][:][int(id)])
                alt.append(h5f['Event_Info/ei_alt'][:][int(id)])
            idx['energy_true'] = np.array(energy_MC)
            idx['alt'] = np.array(alt)
            idx['az'] = np.array(az)
            idx['intensity'] = h5f['LST/intensities'][:]
            idx['leakage2'] = h5f['LST/intensities_width_2'][:]
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
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def get_direction_nominal_frame(self, indexes):

        az_LST = 3.141592654  # rad
        alt_LST = 1.221730476  # rad

        point = AltAz(
            alt=alt_LST * u.rad,
            az=az_LST * u.rad
        )
        # alt = indexes['alt'] # altitude
        # az = indexes['az'] # azimuth
        # print("\nalt shape: {}, az shape: {}".format(indexes['alt'].shape, indexes['az'].shape))
        src = AltAz(
            alt=indexes['alt'] * u.rad,
            az=indexes['az'] * u.rad
        )
        source_direction = src.transform_to(NominalFrame(origin=point))

        return source_direction.delta_alt.deg, source_direction.delta_az.deg

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, 100, 100, self.arrival_time + 1])
        y = np.empty([self.batch_size, self.outcomes], dtype=float)
        print("\ny shape: ", y.shape)

        if self.feature == 'energy':
            y = indexes['energy_true'] # MC energy
        elif self.feature == 'direction':
            #alt_LST, az_LST = 70, 180 # pointing direction of LST (in deg): it's hardcoded, it shouldn't

            # Source direction in Nominal Frame!
            delta_alt, delta_az = self.get_direction_nominal_frame(indexes)
            y[:, 0] = delta_alt
            y[:, 1] = delta_az
        # Generate data
        for i, row in indexes.iterrows():

            filename = self.h5files[int(row['file_index'])]

            h5f = h5py.File(filename, 'r')

            # Store images
            x[i, :, :, 0] = h5f['LST/LST_image_charge_interp'][int(row['image_index'])]
            if self.arrival_time:
                x[i, :, :, 1] = h5f['LST/LST_image_peak_times_interp'][int(row['image_index'])]

            # Store features
            #if self.feature == 'energy':
            #    y[i] = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[2])])
            #elif self.feature == 'direction':
            #if self.feature == 'direction':
            #    # delta az, delta alt computation
            #    y[i, 0] = h5f['LST/delta_alt'][:][int(row[1])]
            #    y[i, 1] = h5f['LST/delta_az'][:][int(row[1])]

            h5f.close()

        # x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y



# generator for the Random Forest
# class DataGeneratorRF(keras.utils.Sequence):
#     'Generates data for Keras'
#
#     def __init__(self, h5files, batch_size=32, shuffle=True):
#         self.batch_size = batch_size
#         self.h5files = h5files
#         self.indexes = np.array([], dtype=np.int64).reshape(0, 4)
#         self.shuffle = shuffle
#         self.generate_indexes()
#         # Load the camera
#         self.geom = CameraGeometry.from_name("LSTCam")
#         self.cleaning_level = {'LSTCam': (3.5, 7.5, 2)}
#         if shuffle: np.random.shuffle(self.indexes)
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         # total number of images in the dataset
#         return int(np.floor(self.indexes.shape[0] / self.batch_size))
#
#     def __getitem__(self, index):
#
#         # print("training idx: ", index, '/', self.__len__())
#
#         # with self.lock:
#         'Generate one batch of data'
#         # index goes from 0 to the number of batches
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         # Find list of IDs
#         # list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         y, energy, altaz, tgradient, hillas, disp = self.__data_generation(indexes)
#
#         # print("training idx: ", indexes)
#
#         return y, energy, altaz, tgradient, hillas, disp
#
#     def get_indexes(self):
#         return self.indexes[0:self.__len__() * self.batch_size]
#
#     def chunkit(self, seq, num):
#
#         avg = len(seq) / float(num)
#         out = []
#         last = 0.0
#
#         while last < len(seq):
#             out.append(seq[int(last):int(last + avg)])
#             last += avg
#
#         return out
#
#     def worker(self, h5files, positions, i, return_dict):
#
#         idx = np.array([], dtype=np.int64).reshape(0, 4)
#
#         for l, f in enumerate(h5files):
#             h5f = h5py.File(f, 'r')
#             lst_idx = h5f['LST/LST_event_index'][:]
#             h5f.close()
#             r = np.arange(len(lst_idx))
#
#             fn_basename = os.path.basename(os.path.normpath(f))
#
#             clas = np.zeros(len(r))  # class: proton by default
#
#             if fn_basename.startswith('g'):
#                 clas = np.ones(len(r))
#
#             cp = np.dstack(([positions[l]] * len(r), r, clas, lst_idx)).reshape(-1, 4)
#
#             idx = np.append(idx, cp, axis=0)
#         return_dict[i] = idx
#
#     def generate_indexes(self):
#
#         cpu_n = multiprocessing.cpu_count()
#         pos = self.chunkit(np.arange(len(self.h5files)), cpu_n)
#         h5f = self.chunkit(self.h5files, cpu_n)
#
#         manager = multiprocessing.Manager()
#         return_dict = manager.dict()
#
#         processes = []
#
#         if cpu_n >= len(self.h5files):
#             # print('ncpus >= num_files')
#             for i, f in enumerate(self.h5files):
#                 p = multiprocessing.Process(target=self.worker, args=([f], [i], i, return_dict))
#                 p.start()
#                 processes.append(p)
#         else:
#             # print('ncpus < num_files')
#             for i in range(cpu_n):
#                 p = multiprocessing.Process(target=self.worker, args=(h5f[i], pos[i], i, return_dict))
#                 p.start()
#                 processes.append(p)
#
#         for p in processes:
#             p.join()
#
#         for key, value in return_dict.items():
#             self.indexes = np.append(self.indexes, value, axis=0)
#
#     def __data_generation(self, indexes):
#
#         'Generates data containing batch_size samples'
#         # Initialization
#         y = np.empty([self.batch_size, 1], dtype=int)
#         energy = np.empty([self.batch_size, 1], dtype=float)
#         altaz = np.empty([self.batch_size, 2], dtype=float)
#         tgradient = np.empty([self.batch_size, 2], dtype=float)
#         hillas = np.empty([self.batch_size, 13], dtype=float)
#         disp = np.empty([self.batch_size, 2], dtype=float)
#
#         boundary, picture, min_neighbors = self.cleaning_level['LSTCam']
#
#         point = AltAz(alt=70 * u.deg, az=0 * u.deg)
#
#         # Generate data
#         for i, row in enumerate(indexes):
#             filename = self.h5files[int(row[0])]
#
#             h5f = h5py.File(filename, 'r')
#
#             charge = h5f['LST/LST_image_charge'][int(row[1])]
#             energy[i] = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[3])])
#
#             # Apply image cleaning
#             clean = tailcuts_clean(
#                 self.geom,
#                 charge,
#                 boundary_thresh=boundary,
#                 picture_thresh=picture,
#                 min_number_picture_neighbors=min_neighbors
#             )
#
#             h = hillas_parameters(self.geom[clean], charge[clean])
#
#             l = leakage(self.geom, charge, clean)
#
#             n_islands, _ = number_of_islands(self.geom, clean)
#
#             hillas[i, 0] = h['intensity']
#             hillas[i, 1] = h['width'] / u.m
#             hillas[i, 2] = h['length'] / u.m
#             hillas[i, 3] = hillas[i, 1] / hillas[i, 2]  # h['wl']
#             hillas[i, 4] = h['phi'] / u.rad
#             hillas[i, 5] = h['psi'] / u.rad
#             hillas[i, 6] = h['skewness']
#             hillas[i, 7] = h['kurtosis']
#             hillas[i, 8] = h['r'] / u.m
#             hillas[i, 9] = l['leakage2_intensity']
#             hillas[i, 10] = n_islands
#             hillas[i, 11] = h['x'] / u.m
#             hillas[i, 12] = h['y'] / u.m
#
#             altaz[i, 0] = h5f['LST/delta_alt'][:][int(row[1])]
#             altaz[i, 1] = h5f['LST/delta_az'][:][int(row[1])]
#
#             time = h5f['LST/LST_image_peak_times'][int(row[1])]
#
#             timing = timing_parameters(
#                 self.geom,
#                 image=charge,
#                 peakpos=time,
#                 hillas_parameters=h,
#             )
#
#             tgradient[i, 0] = timing['slope'] * u.m
#             tgradient[i, 1] = timing['intercept']
#
#             src = NominalFrame(origin=point, delta_az=altaz[i, 1] * u.deg, delta_alt=altaz[i, 0] * u.deg)
#             source_direction = src.transform_to(AltAz)
#
#             src_pos = sky_to_camera(source_direction.alt,
#                                     source_direction.az,
#                                     28 * u.m,
#                                     70 * u.deg,
#                                     0 * u.deg)
#
#             disp_container = disp_parameters(hillas=h, source_pos_x=src_pos.x, source_pos_y=src_pos.y)
#
#             disp[i, 0] = disp_container.dx / u.m
#             disp[i, 1] = disp_container.dy / u.m
#
#             y[i] = int(row[2])
#
#             h5f.close()
#
#         return y, energy, altaz, tgradient, hillas, disp


class DataGeneratorChain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, batch_size=32, arrival_time=False, shuffle=True, intensity=50, leakage2_intensity=0.2):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 6)
        self.shuffle = shuffle
        self.arrival_time = arrival_time
        self.generate_indexes()
        # Load the camera
        self.intensity = intensity
        self.leakage2_intensity = leakage2_intensity
 #       self.apply_cuts()
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):

        # print("training idx: ", index, '/', self.__len__())

        # with self.lock:
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y, intensity, energy, altaz = self.__data_generation(indexes)

        # print("training idx: ", indexes)

        return x, y, intensity, energy, altaz

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

#    def apply_cuts(self):
#        self.indexes = self.indexes[
#            (self.indexes[:, 4] > self.intensity) & (self.indexes[:, 5] < self.leakage2_intensity)]

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 6)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            lst_idx = h5f['LST/LST_event_index'][:]
            intensities = h5f['LST/intensities'][:]
            intensities_width_2 = h5f['LST/intensities_width_2'][:]
            h5f.close()
            r = np.arange(len(lst_idx))

            fn_basename = os.path.basename(os.path.normpath(f))

            clas = np.zeros(len(r))  # class: proton by default

            if fn_basename.startswith('g'):
                clas = np.ones(len(r))

            cp = np.dstack(([positions[l]] * len(r), r, clas, lst_idx, intensities, intensities_width_2)).reshape(-1, 6)

            idx = np.append(idx, cp, axis=0)

        return_dict[i] = idx

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
            self.indexes = np.append(self.indexes, value, axis=0)

    def __data_generation(self, indexes):

        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, 100, 100, self.arrival_time + 1])
        y = np.empty([self.batch_size, 1], dtype=int)
        intensity = np.empty([self.batch_size, 1], dtype=float)
        energy = np.empty([self.batch_size, 1], dtype=float)
        altaz = np.empty([self.batch_size, 2], dtype=float)

        # Generate data
        for i, row in enumerate(indexes):
            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')

            # Store image
            x[i, :, :, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, :, :, 1] = h5f['LST/LST_image_peak_times_interp'][int(row[1])]

            energy[i] = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[3])])

            intensity[i] = h5f['LST/intensities'][:][int(row[1])]

            altaz[i, 0] = h5f['LST/delta_alt'][:][int(row[1])]
            altaz[i, 1] = h5f['LST/delta_az'][:][int(row[1])]

            y[i] = int(row[2])

            h5f.close()

        return x, y, intensity, energy, altaz
