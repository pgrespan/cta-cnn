import argparse
import multiprocessing as mp
from multiprocessing import Process
from os import listdir, remove
from os.path import isfile, join

import h5py
import numpy as np
import tables
from astropy import units as u
from ctapipe.coordinates import NominalFrame, AltAz
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.instrument import CameraGeometry
from scipy.interpolate import griddata
from tables.exceptions import HDF5ExtError, NoSuchNodeError
from tqdm import tqdm, trange

'''
usage: python lst_interpolate.py --dirs path/to/folder1 path/to/folder2 ... --rem_org 0 --rem_corr 0 --rem_nsnerr 0
'''


def get_event_data(data):
    data_einfo = data.root.Events

    # event info data
    ei_alt = [x['alt'] for x in data_einfo.iterrows()]
    ei_az = [x['az'] for x in data_einfo.iterrows()]
    ei_mc_energy = [x['mc_energy'] for x in data_einfo.iterrows()]

    return data_einfo, ei_alt, ei_az, ei_mc_energy


def get_LST_data(data):
    data_LST = data.root.LST_LSTCam

    # LST data
    # MEMO if [i, 0, j, k] is e.g. the list #3, then ith, jth and kth belong to event #3
    LST_event_index = [x['event_index'] for x in data_LST.iterrows()]
    LST_image_charge = [x['charge'] for x in data_LST.iterrows()]
    LST_image_peak_times = [x['peakpos'] for x in data_LST.iterrows()]

    return data_LST, LST_event_index, LST_image_charge, LST_image_peak_times


def func(paths, ro, rc, rn):
    # img_rows, img_cols = 100, 100

    # iterate on each proton file & concatenate charge arrays
    for f in paths:

        # get the data from the file
        try:
            data_p = tables.open_file(f)
            _, LST_event_index, LST_image_charge, LST_image_peak_times = get_LST_data(data_p) #OK
            _, ei_alt, ei_az, ei_mc_energy = get_event_data(data_p) #OK

            # Excluding the 0th element - it's the empty one!!
            LST_event_index = LST_event_index[1:]
            LST_image_charge = LST_image_charge[1:]
            LST_image_peak_times = LST_image_peak_times[1:]

            # get camera geometry & camera pixels coordinates
            camera = CameraGeometry.from_name("LSTCam")
            points = np.array([np.array(camera.pix_x / u.m), np.array(camera.pix_y / u.m)]).T #OK

            grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]

            # alt az of the array (CTA?)
            az_array = 0            # before was ai_run_array_direction[0][0], now it is hardcoded as it's not present in new files
            alt_array = 1.2217305   # ai_run_array_direction[0][1]

            '''
            was probably useless and wrong even with the old simulations
            if alt_array > 90:
                alt_array = 90
            '''

            # LST coordinates (pointing position)
            point = AltAz(alt=alt_array * u.rad, az=az_array * u.rad)

            lst_image_charge_interp = []
            lst_image_peak_times_interp = []
            delta_az = []
            delta_alt = []
            intensities = []
            intensities_width_2 = []
            acc_idxs = []  # accepted indexes    # in principle it can be removed when cuts (line AAA) are not used here

            cleaning_level = {'LSTCam': (3.5, 7.5, 2)}
            count = 0
            for i in trange(0, len(LST_image_charge)):

                image = LST_image_charge[i]
                time = LST_image_peak_times[i]

                boundary, picture, min_neighbors = cleaning_level['LSTCam']
                clean = tailcuts_clean(
                    camera,
                    image,
                    boundary_thresh=boundary,
                    picture_thresh=picture,
                    min_number_picture_neighbors=min_neighbors
                )

                if len(np.where(clean > 0)[0]) != 0:
                    hillas = hillas_parameters(camera[clean], image[clean])
                    intensity = hillas['intensity']

                    l = leakage(camera, image, clean)
                    # print(l)
                    leakage2_intensity = l['intensity_width_2']

                    # if intensity > 50 and leakage2_intensity < 0.2:  # ------>AAA --- CUT DIRECTLY DURING INTERP
                    # cubic interpolation
                    interp_img = griddata(points, image, (grid_x, grid_y), fill_value=0, method='cubic')
                    interp_time = griddata(points, time, (grid_x, grid_y), fill_value=0, method='cubic')

                    # delta az, delta alt computation
                    az = ei_az[LST_event_index[i]]
                    alt = ei_alt[LST_event_index[i]]
                    src = AltAz(alt=alt * u.rad, az=az * u.rad)
                    source_direction = src.transform_to(NominalFrame(origin=point))

                    # appending to arrays
                    lst_image_charge_interp.append(interp_img)
                    lst_image_peak_times_interp.append(interp_time)
                    delta_az.append(source_direction.delta_az.deg)
                    delta_alt.append(source_direction.delta_alt.deg)

                    intensities.append(intensity)
                    intensities_width_2.append(leakage2_intensity)

                    acc_idxs += [i]  # also this one can be removed when no cuts here

                else:
                    count += 1
                    print("Event #{} rejected (No islands)! Cumulative count: {}".format(i, count))

            # lst_image_charge_interp = np.array(lst_image_charge_interp)

            data_p.close()

            filename = f[:-3] + '_interp.h5'

            print("Writing file: " + filename)

            data_file = h5py.File(filename, 'w')

            data_file.create_dataset('Event_Info/ei_alt', data=np.array(ei_alt))
            data_file.create_dataset('Event_Info/ei_az', data=np.array(ei_az))
            data_file.create_dataset('Event_Info/ei_mc_energy', data=np.array(ei_mc_energy))

            data_file.create_dataset('LST/LST_event_index', data=np.array(LST_event_index)[acc_idxs])
            data_file.create_dataset('LST/LST_image_charge', data=np.array(LST_image_charge)[acc_idxs])
            data_file.create_dataset('LST/LST_image_peak_times', data=np.array(LST_image_peak_times)[acc_idxs])
            #data_file.create_dataset('LST/LST_event_index', data=np.array(LST_event_index))
            #data_file.create_dataset('LST/LST_image_charge', data=np.array(LST_image_charge))
            #data_file.create_dataset('LST/LST_image_peak_times', data=np.array(LST_image_peak_times))
            data_file.create_dataset('LST/LST_image_charge_interp', data=np.array(lst_image_charge_interp))
            data_file.create_dataset('LST/LST_image_peak_times_interp', data=np.array(lst_image_peak_times_interp))
            data_file.create_dataset('LST/delta_alt', data=np.array(delta_alt))
            data_file.create_dataset('LST/delta_az', data=np.array(delta_az))

            data_file.create_dataset('LST/intensities', data=np.array(intensities))
            data_file.create_dataset('LST/intensities_width_2', data=np.array(intensities_width_2))

            data_file.close()

            # in the interpolated files there will be all the original events
            # but for the LST only the ones actually seen at least from one LST (as in the original files)
            # and that are above thresholds cuts

            if ro == '1':
                remove(f)
                print('Removing original file')

        except HDF5ExtError:

            print('\nUnable to open file' + f)

            if rc == '1':
                print('Removing it...')
                remove(f)

        except NoSuchNodeError:

            print('This file has a problem with the data structure: ' + f)

            if rn == '1':
                print('Removing it...')
                remove(f)


def chunkit(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files.', required=True)
    parser.add_argument(
        '--rem_org', type=str, default='0', help='Select 1 to remove the original files.', required=False)
    parser.add_argument(
        '--rem_corr', type=str, default='0', help='Select 1 to remove corrupted files.', required=False)
    parser.add_argument(
        '--rem_nsnerr', type=str, default='0', help='Select 1 to remove files that raise NoSuchNodeError exception.',
        required=False)
    parser.add_argument(
        '--parallel', type=bool, default=False,
        help='Select 0 to process files using single CPU computation (useful when files are too big).',
        required=False)

    FLAGS, unparsed = parser.parse_known_args()

#    print(FLAGS.dirs)

    ncpus = mp.cpu_count()
    print("\nNumber of CPUs: " + str(ncpus))

    # get all the parameters given by the command line
    folders = FLAGS.dirs

    print('Folders: ' + str(folders) + '\n')

    # create a single big list containing the paths of all the files
    all_files = []

    for path in folders:
        files = [join(path, f) for f in listdir(path) if (
                isfile(join(path, f))
                and f.endswith(".h5")
                and not f.endswith("_interp.h5")
                and (f[:-3]+'_interp.h5') not in listdir(path))]
        all_files = all_files + files

    # print('Files: ' + '\n' + str(all_files) + '\n')

    num_files = len(all_files)

    print('num_files: ', num_files)

    if FLAGS.parallel:

        processes = []

        if ncpus >= num_files:
            print('ncpus >= num_files')
            for f in all_files:
                p = Process(target=func, args=([f], FLAGS.rem_org, FLAGS.rem_corr, FLAGS.rem_nsnerr))
                p.start()
                processes.append(p)
        else:
            print('ncpus < num_files')
            c = chunkit(all_files, ncpus)
            for f in c:
                p = Process(target=func, args=(f, FLAGS.rem_org, FLAGS.rem_corr, FLAGS.rem_nsnerr))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

    else:

        func(all_files, FLAGS.rem_org, FLAGS.rem_corr, FLAGS.rem_nsnerr)
